# _epoch_whole_distill是蒸馏的函数、train_whole_on_udds是训练的函数
import torch
from model import MLPStudentEncoder

def _epoch_whole_distill(model, loader: DataLoader, optimizer=None, device=DEVICE, desc="",
                         studentA=None, studentB=None, kd_weight: float = 0.5):
    printed_kd = False
    training = optimizer is not None
    model.train(training)
    if studentA is not None:
        studentA.train(training)
    if studentB is not None:
        studentB.train(training)

    tot, n = 0.0, 0
    ys, yps, times_all, cids_all = [], [], [], []

    for xA, xB, y, t_end, *rest in tqdm(loader, desc=desc, leave=False):
        xA = xA.to(device)
        xB = xB.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        sA = studentA(xA)
        sB = studentB(xB)
        _, _, sA2, sB2 = model.star(sA, sB)
        y_student = model.head(torch.cat([sA2, sB2], dim=-1))  # [B]

        y_std = y.detach().std(unbiased=False).clamp_min(5e-2)
        resid = (y_student - y) / y_std
        sup_loss = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=0.5)

        with torch.no_grad():
            tA = model.encoder_a(xA)
            tB = model.encoder_b(xB)

        def _l2_normalize(x, eps=1e-6):
            return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))

        tA_n = _l2_normalize(tA)
        tB_n = _l2_normalize(tB)
        sA_n = _l2_normalize(sA)
        sB_n = _l2_normalize(sB)

        kdA = F.mse_loss(sA_n, tA_n)
        kdB = F.mse_loss(sB_n, tB_n)

        loss = sup_loss + kd_weight * (kdA + kdB)

        if training and not printed_kd:
            print(f"[KD Debug] sup={float(sup_loss):.4f} kdA={float(kdA):.4f} kdB={float(kdB):.4f}")
            printed_kd = True

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) +
                                           list(studentA.parameters()) +
                                           list(studentB.parameters()), 1.0)
            optimizer.step()

        bs = y.size(0)
        tot += float(loss.item()) * bs
        n += bs

        ys.append(y.detach().cpu().numpy().reshape(-1))
        yps.append(y_student.detach().cpu().numpy().reshape(-1))

        times_all.append(t_end.detach().cpu().numpy().reshape(-1))

        cids = rest[0]
        cids_all.append(cids.detach().cpu().numpy().reshape(-1))

    y = np.concatenate(ys) if ys else np.array([])
    yp = np.concatenate(yps) if yps else np.array([])
    t = np.concatenate(times_all) if times_all else None
    cid = np.concatenate(cids_all) if cids_all  else None
    return tot / max(1, n), y, yp, t, cid

def train_whole_on_udds(device=DEVICE,
                        save_best_path: str = None,
                        ckpt_dir: str = None,
                        resume: bool = True,
                        keep_every_epoch: bool = False,
                        loaders=None):
    if loaders is None:
        ltr, lva, lte = build_udds_dataloaders_from_dir(
            UDDS_DIR, WINDOW_UDDS, STRIDE_UDDS,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
    else:
        ltr, lva, lte = loaders

    model = build_whole_model().to(device)
    model_name = type(model).__name__
    tag = f"[{model_name}|WHOLE]"

    studentA = studentB = None
    if DISTILL_ENABLE:
        dimA = int(model.encoder_a.out_dim)
        dimB = int(model.encoder_b.out_dim)
        studentA = MLPStudentEncoder(in_dim=3, hidden_size=dimA).to(device)
        studentB = MLPStudentEncoder(in_dim=3, hidden_size=dimB).to(device)
        print(f"{tag} [KD] Enable distill: A_dim={dimA}, B_dim={dimB}, weight={DISTILL_WEIGHT}")

    base_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if DISTILL_ENABLE:
        opt_params = base_params + list(studentA.parameters()) + list(studentB.parameters())
    else:
        opt_params = base_params

    optimizer = torch.optim.AdamW(opt_params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=5e-5
    )

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt") if ckpt_dir else None
    if save_best_path is None and ckpt_dir:
        save_best_path = os.path.join(ckpt_dir, "best.pt")

    start_ep = 1
    best_val = float("inf")
    bad = 0
    if resume and last_path and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"]:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception:
                    pass
            start_ep = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", float("inf")))
            bad = int(ckpt.get("bad", 0))
            print(f"{tag} [Resume] epoch={start_ep-1}, best_val={best_val:.6f}, bad={bad}")
        except Exception as e:
            print(f"{tag} [Resume Failed] {e}，将从头训练。")

    for ep in range(start_ep, EPOCHS + 1):
        if DISTILL_ENABLE:
            tr_loss, _, _, _, _ = _epoch_whole_distill(
                model, ltr, optimizer, device,
                desc=f"{tag} Train ep{ep}",
                studentA=studentA, studentB=studentB, kd_weight=DISTILL_WEIGHT
            )
            va_loss, _, _, _, _ = _epoch_whole_distill(
                model, lva, None, device,
                desc=f"{tag}  Val  ep{ep}",
                studentA=studentA, studentB=studentB, kd_weight=DISTILL_WEIGHT
            )
        else:
            tr_loss, _, _, _, _ = _epoch_whole(model, ltr, optimizer, device, desc=f"{tag} Train ep{ep}")
            va_loss, _, _, _, _ = _epoch_whole(model, lva, None,      device, desc=f"{tag}  Val  ep{ep}")

        print(f"{tag} ep {ep:02d}: train {tr_loss:.8f} | val {va_loss:.8f} | patience {bad}/{PATIENCE}")
        scheduler.step(va_loss)

        improved = va_loss < best_val - 1e-12
        if improved:
            best_val = va_loss
            bad = 0
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
                torch.save(model.state_dict(), save_best_path)
        else:
            bad += 1

        if ckpt_dir:
            state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "bad": bad,
                "network": type(model).__name__,
                "distill": bool(DISTILL_ENABLE),
            }
            torch.save(state, last_path)
            if keep_every_epoch:
                torch.save(state, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt"))

        if bad >= PATIENCE:
            print(f"{tag} Early stop."); break

    if save_best_path and os.path.isfile(save_best_path):
        model.load_state_dict(torch.load(save_best_path, map_location=device))

    if DISTILL_ENABLE:
        _, y_true, y_pred, times, cids = _epoch_whole_distill(
            model, lte, None, device, desc=f"{tag}  Test",
            studentA=studentA, studentB=studentB, kd_weight=DISTILL_WEIGHT
        )
    else:
        _, y_true, y_pred, times, cids = _epoch_whole(model, lte, None, device, desc=f"{tag}  Test")

    return y_true, y_pred, times, cids


