import os
import torch
from torch import nn
import yaml
from torch.cuda.amp import GradScaler
from torch import autocast


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# random weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def fix_state_dict(state_dict):
    return {
        ".".join(k.split(".")[1:]) if k.startswith("module") else k: v
        for k, v in state_dict.items()
    }


def load_state(model, checkpoint, device, net_load_disc=None):
    checkpoint = torch.load(checkpoint, map_location=device) if checkpoint else None
    net_load_disc = (
        torch.load(net_load_disc, map_location=device) if net_load_disc else None
    )
    return model.load_state(checkpoint, device, net_load_disc=net_load_disc)


class Scheduler:
    def __init__(self, optimizer):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10
        )

    def update(self):
        self.lr_scheduler.step()


class ScalerWrapper:
    def __init__(self, model, optimizer, loss, scale_mode="fp32"):
        self.scaler = None if scale_mode == "fp32" else GradScaler()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_fn = self.get_fn(model.do_train_batch)
        self.test_fn = self.get_fn(model.do_test_batch)

    def get_fn(self, fn):
        def run_fn(batch):
            if self.scaler is not None:
                inputs = self.model.parse_batch(batch)
                model_inputs, label = inputs[:-1], inputs[-1]
                with autocast(device_type="cuda", dtype=torch.float16):
                    return fn(model_inputs, label, self.loss)
            return fn(batch)

        return run_fn

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def get_model(self):
        return self.model

    def do_train(self, batch):
        self.optimizer.zero_grad()
        out, Loss = self.train_fn(batch)
        self.backward(Loss)
        self.step()
        self.update()
        return out

    def init_iter(self, epoch_num, res):
        self.model.init_iter(epoch_num, res)

    def load_optimizer(self, *inputs):
        self.optimizer.load(*inputs)

    def update_optimizer(self):
        self.optimizer.update()

    def do_test(self, batch, probs):
        out = self.test_fn(batch)
        row = self.model.get_results_row(batch, out)
        return [torch.cat((probs[r], row[r]), dim=0) for r in range(len(row))]

    def calc_acc(self, probs):
        return self.model.calc_acc(probs)

    def get_init_probs(self):
        return self.model.get_init_probs()

    def best_epoch_metric(self, epoch_num):
        return self.model.best_epoch_metric(epoch_num)

    def backward(self, loss):
        if self.scaler is None:
            loss.backward()
            return
        self.scaler.scale(loss).backward()

    def step(self):
        if self.scaler is None:
            self.optimizer.step()
            return
        idx = range(len(self.optimizer.lr_scheduler))
        for s in idx:
            self.scaler.step(self.optimizer.optimizer[s])

    def update(self):
        if self.scaler is not None:
            self.scaler.update()


class OptimizerWrapper:
    def __init__(self, model, lr):
        self.optimizer = model.get_opt(lr)
        self.lr_scheduler = [Scheduler(opt) for opt in self.optimizer]

    def update(self, idx=None):
        idx = range(len(self.lr_scheduler)) if idx is None else [idx]
        for s in idx:
            self.lr_scheduler[s].update()

    def zero_grad(self, idx=None):
        idx = range(len(self.lr_scheduler)) if idx is None else [idx]
        for s in idx:
            self.optimizer[s].zero_grad()

    def step(self, idx=None):
        idx = range(len(self.lr_scheduler)) if idx is None else [idx]
        for s in idx:
            self.optimizer[s].step()

    def load(self, itrs, num_batch):
        for epoch_num in range(itrs):
            for t in range(num_batch):
                self.zero_grad()
                self.step()
            self.update()


def resume_state(
    model, path, results_f, lr, device, gen=False, mode="asa", flip_label=False
):
    if gen:
        if path is None and os.path.exists(
            os.path.join("/".join(results_f.split("/")[:-1]), "conf.yaml")
        ):
            os.unlink(os.path.join("/".join(results_f.split("/")[:-1]), "conf.yaml"))
        if not os.path.exists(
            os.path.join("/".join(results_f.split("/")[:-1]), "conf.yaml")
        ):
            with open(
                os.path.join("/".join(results_f.split("/")[:-1]), "conf.yaml"),
                "w",
                encoding="utf-8",
            ) as yaml_file:
                dump = yaml.dump(
                    {"mode": mode, "flip_label": flip_label},
                    default_flow_style=False,
                    allow_unicode=True,
                    encoding=None,
                )
                yaml_file.write(dump)
        with open(
            os.path.join("/".join(results_f.split("/")[:-1]), "conf.yaml"), "r"
        ) as f:
            conf = yaml.load(f, Loader=yaml.Loader)

    if path is None:
        return (
            model,
            OptimizerWrapper(model, lr),
            0,
            None,
            conf["mode"] if gen else None,
            conf["flip_label"] if gen else None,
        )

    epoch = path.split("/")[-1].split(".")[-2]
    epoch = epoch[epoch.find("model") + 5 :]
    epoch = int(epoch) if len(epoch) > 0 else None
    with open(results_f, "r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
    idx = results["best"]
    epoch_res = results["epochs"][epoch] if epoch is not None else best_res
    best_res = epoch_res
    if "factor" in best_res.keys():
        try:
            model.module.factor = epoch_res["factor"]
            model.module.no_disc_opt = results["no_disc_opt"]
            model.module.start_adv = results.get("start_adv", 5)
        except:
            model.factor = 2  # epoch_res["factor"]
            model.no_disc_opt = results["no_disc_opt"]
            model.start_adv = results.get("start_adv", 5)

    best_res = best_res["result"]
    last = results["last"]
    name = os.path.basename(path)
    itrs = idx + 1 if epoch is None else epoch + 1
    optimizer = OptimizerWrapper(model, lr)
    for i in range(itrs, last + 1):
        del results["epochs"][i]
    results["last"] = itrs - 1

    with open(results_f, "w", encoding="utf-8") as yaml_file:
        dump = yaml.dump(
            results,
            default_flow_style=False,
            allow_unicode=True,
            encoding=None,
        )
        yaml_file.write(dump)

    return (
        model,
        optimizer,
        itrs,
        best_res,
        conf["mode"] if gen else None,
        conf["flip_label"] if gen else None,
    )


def log_epoch(
    model,
    res,
    res_trap,
    epoch_num,
    is_best,
    best_res,
    out_f,
    best_res_trap=None,
):
    if os.path.exists(os.path.join(out_f, "log.yaml")):
        with open(os.path.join(out_f, "log.yaml"), "r") as f:
            results = yaml.load(f, Loader=yaml.Loader)
    else:
        results = {}
        results["epochs"] = {}

    out_res = res if res_trap is None else (res, res_trap)
    results["last"] = epoch_num
    out_res = {"result": out_res}
    if hasattr(model, "factor"):
        out_res["factor"] = model.factor
        # results["no_disc_opt"] = model.no_disc_opt
        results["start_adv"] = model.start_adv

    results["epochs"][epoch_num] = out_res
    if is_best:
        results["best"] = epoch_num
    with open(os.path.join(out_f, "log.yaml"), "w", encoding="utf-8") as yaml_file:
        dump = yaml.dump(
            results,
            default_flow_style=False,
            allow_unicode=True,
            encoding=None,
        )
        yaml_file.write(dump)

    Message = "\nEpoch: " + str(epoch_num) + ", Value: " + str(res)
    if best_res_trap is not None:
        Message = Message + ", value_trap: " + str(res_trap)
    Message = Message + ", best: " + str(best_res)
    if best_res_trap is not None:
        Message = Message + ", best_trap: " + str(best_res_trap)
    if is_best:
        model.save_state(os.path.join(out_f, "model.pth"))
    model.save_state(os.path.join(out_f, "model" + str(epoch_num) + ".pth"))
    return Message
