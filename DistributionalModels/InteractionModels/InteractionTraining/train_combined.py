# train combined
import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from file_management import save_to_pickle, load_from_pickle
from Networks.network import ConstantNorm, pytorch_model
from tianshou.data import Collector, Batch, ReplayBuffer
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import run_optimizer, get_weights, get_targets
from DistributionalModels.InteractionModels.InteractionTraining.compute_errors import assess_losses, get_interaction_vals
from Rollouts.rollouts import ObjDict, merge_rollouts

def _combined_logging(full_model, train_args, rollouts, test_rollout, i, batchvals,
                     interaction_likelihood, interaction_binaries, true_binaries,
                     prediction_params, passive_prediction_params,
                     target, active_l2, passive_l2, done_flags, interaction_schedule,
                     forward_error, forward_loss, passive_error, trace):
    # print(i, ": pl: ", pytorch_model.unwrap(passive_error.norm()), " fl: ", pytorch_model.unwrap(forward_loss.norm()), 
    #     " il: ", pytorch_model.unwrap(interaction_loss.norm()), " dl: ", pytorch_model.unwrap(interaction_diversity_loss.norm()))
    print(i, i % (train_args.log_interval * 10), (train_args.log_interval * 10))
    # error
    if i % (train_args.log_interval * 10) == 0 and i != 0:
        print("assessing full train")
        assess_losses(full_model, rollouts)
        # assess_error(rollouts, passive_error_cutoff=train_args.passive_error_cutoff)
        print("assessing test rollouts")
        assess_losses(full_model, test_rollout)
        # assess_error(test_rollout, passive_error_cutoff=train_args.passive_error_cutoff)
    if full_model.multi_instanced: 
        split_target = full_model.split_instances(target)
        inp = full_model.delta(batchvals.values.state)
        inp = full_model.split_instances(inp)
        first_obj = full_model.gamma(batchvals.values.state)[...,:full_model.first_obj_dim]
        active = full_model.split_instances(full_model.rv(prediction_params[0])).squeeze()
        activeunnorm = full_model.split_instances(prediction_params[0]).squeeze()
        activevar = full_model.split_instances(prediction_params[1]).squeeze()
        print(full_model.rv(split_target).shape, full_model.rv(target).shape, full_model.rv(prediction_params[0]).shape)
        adiff = full_model.split_instances(full_model.rv(target) - full_model.rv(prediction_params[0]))
        pdiff = full_model.split_instances(full_model.rv(target) - full_model.rv(passive_prediction_params[0]))

        obj_indices = pytorch_model.unwrap((trace[idxes] > 0).nonzero())
        all_indices = []
        for ti in obj_indices:
            # print(full_model.rv(passive_prediction_params[0])[ti[0]])
            all_indices.append(np.array([ti[0], ti[1]-2]))
            all_indices.append(np.array([ti[0], ti[1]-1]))
            all_indices.append(pytorch_model.unwrap(ti))
            if ti[1]+1 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+1]))
            if ti[1]+2 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+2]))
            for _ in range(3):
                all_indices.append(np.array([ti[0], np.random.randint(interaction_likelihood[ti[0]].shape[0])]))
        for _ in range(20):
            all_indices.append(np.array([np.random.randint(train_args.batch_size), np.random.randint(interaction_likelihood.shape[1])]))

        obj_indices = np.array(all_indices)
        print("Iters: ", i)
        # print(trw[idxes], trace[idxes].sum(dim=1), idxes)
        print(forward_loss[obj_indices[0][0]], interaction_likelihood[obj_indices[0][0]])
        for a in obj_indices[:20]:
            print("idx", a[0], a[1])
            # print("has", trace[idxes[a[0]].sum())
            print("trace: ", pytorch_model.unwrap(trace[idxes[a[0]], a[1]]),
            "inter: ", pytorch_model.unwrap(interaction_likelihood[a[0], a[1]]),
            "first: ", pytorch_model.unwrap(first_obj[a[0]]),
            "inp: ", pytorch_model.unwrap(inp[a[0], a[1]]),
            "target: ", pytorch_model.unwrap(split_target[a[0], a[1]]),
            "active: ", pytorch_model.unwrap(active[a[0], a[1]]),
            "aun: ", pytorch_model.unwrap(activeunnorm[a[0], a[1]]),
            "avar: ", pytorch_model.unwrap(activevar[a[0], a[1]]),
            "adiff: ", pytorch_model.unwrap(adiff[a[0], a[1]]),
            "pdiff: ", pytorch_model.unwrap(pdiff[a[0], a[1]]),
            "floss: ", pytorch_model.unwrap(forward_loss[a[0], a[1]]))
    else:
        likelihood_binaries = (interaction_likelihood > .5).float()
        if train_args.interaction_iters > 0:
            test_binaries = (interaction_binaries.squeeze() + true_binaries.squeeze() + likelihood_binaries.squeeze()).long().squeeze()
            test_idxes = torch.nonzero(test_binaries)
            # print(test_idxes.shape, test_binaries.shape)
            intbint = torch.cat([interaction_likelihood, likelihood_binaries, interaction_binaries, true_binaries, forward_error, passive_error], dim=1).squeeze()
        else:
            passive_binaries = (passive_l2 > train_args.passive_error_cutoff).float().unsqueeze(1)
            test_binaries = (interaction_binaries.squeeze() + likelihood_binaries.squeeze())# + passive_binaries.squeeze())
            print([interaction_likelihood.shape, likelihood_binaries.shape, interaction_binaries.shape, true_binaries.shape, test_binaries.shape, forward_error.shape, passive_error.shape])
            if train_args.compare_trace: test_binaries += true_binaries.squeeze()
            test_binaries = test_binaries.long().squeeze()
            test_idxes = torch.nonzero(test_binaries)
            if train_args.compare_trace: intbint = torch.cat([interaction_likelihood, likelihood_binaries, interaction_binaries, true_binaries, passive_binaries, forward_error, passive_error], dim=1).squeeze()
            else: intbint = torch.cat([interaction_likelihood, likelihood_binaries, interaction_binaries, passive_binaries, forward_error, passive_error], dim=1).squeeze()
        test_binaries[test_binaries > 1] = 1
        inp = torch.cat([full_model.gamma(batchvals.values.state), full_model.delta(get_targets(full_model.predict_dynamics, batchvals))], dim=-1)
        # print(inp.shape, batchvals.values.state.shape, prediction_params[0].shape, test_idxes, test_binaries, likelihood_binaries, interaction_binaries)
        # print(test_idxes)

        # choose only test indexes from the selection indexes
        # select_indices = np.concatenate([np.arange(1,15), np.arange(train_args.batch_size-15,train_args.batch_size)]).squeeze() if train_args.batch_size > 30 else np.arange(0, train_args.batch_size)
        # test_idxes = torch.tensor([k for k,i in enumerate(select_indices) if i in test_idxes.squeeze()]).long()
        
        # test_idxes = torch.tensor([i[0] for i in test_idxes if i in select_indices]).long()
        print(test_idxes)
        print("Iters: ", i,
            "input", inp[test_idxes].squeeze(),
            # "\ninteraction", interaction_likelihood,
            # "\nbinaries", interaction_binaries[select_indices],
            # "\ntrue binaries", true_binaries,
            "\nintbint", intbint[test_idxes].squeeze(),
            # "\naoutput", full_model.rv(prediction_params[0]),
            # "\navariance", full_model.rv(prediction_params[1]),
            # "\npoutput", full_model.rv(passive_prediction_params[0]),
            # "\npvariance", full_model.rv(passive_prediction_params[1]),
            # full_model.delta(batchvals.values.next_state[0]), 
            # full_model.gamma(batchvals.values.state[0]),
            "\ntarget: ", full_model.rv(target)[test_idxes].squeeze(),
            "\nactive", full_model.rv(prediction_params[0])[test_idxes].squeeze(),
            # "\ntadiff", (full_model.rv(target) - full_model.rv(prediction_params[0])) * test_binaries,
            # "\ntpdiff", (full_model.rv(target) - full_model.rv(passive_prediction_params[0])) * test_binaries,
            "\ntadiff", (full_model.rv(target) - full_model.rv(prediction_params[0]))[test_idxes].squeeze(),
            "\ntpdiff", (full_model.rv(target) - full_model.rv(passive_prediction_params[0]))[test_idxes].squeeze(),
            "\nal2: ", active_l2.mean(dim=0),
            "\npl2: ", passive_l2.mean(dim=0),)
    print(
        "\nae: ", (forward_error * done_flags).mean(dim=0),
        "\nal: ", (forward_loss * done_flags).sum(dim=0) / interaction_likelihood.sum(),
        "\npl: ", (passive_error * done_flags).mean(dim=0),
        "\ninter_lambda: ", interaction_schedule(i),
        "\ncombined compute mem", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))


def _train_combined_interaction(full_model, train_args, rollouts, idxes_sets, half_batchvals, batchvals, proximal, use_weights, alt_weights, inter_loss, interaction_optimizer):
    # resamples because the interaction weights should be based on different values
    # TODO: only samples half and half at the moment
    if idxes_sets is None: idxes_noweight, batchvals_noweight = rollouts.get_batch(train_args.batch_size // 2, weights=None, existing=half_batchvals[0])
    else: idxes_noweight, batchvals_noweight = rollouts.get_batch(train_args.batch_size // 2, existing=half_batchvals[0], idxes=idxes_sets[i])
    # inter_weights = proximal if proximal is not None else use_weights
    if train_args.interaction_boosting > 0: use_weights = alt_weights if np.random.rand() < train_args.interaction_boosting else use_weights
    if idxes_sets is None: idxes_weight, batchvals_weight = rollouts.get_batch(train_args.batch_size // 2, weights=use_weights, existing=half_batchvals[1])
    else: idxes_weight, batchvals_weight = rollouts.get_batch(train_args.batch_size // 2, existing=half_batchvals[1], idxes=idxes_sets[i])
    idxes =  np.concatenate([idxes_noweight, idxes_weight])
    batchvals = merge_rollouts([batchvals_noweight, batchvals_weight], existing=batchvals)
    # TODO: copied from above, but it would be nice to separate as a separate function
    prediction_params = full_model.forward_model(full_model.gamma(batchvals.values.state))
    if full_model.multi_instanced: interaction_likelihood = full_model.interaction_model.instance_labels(full_model.gamma(batchvals.values.state))
    else: interaction_likelihood = full_model.interaction_model(full_model.gamma(batchvals.values.state))
    passive_prediction_params = full_model.passive_model(full_model.delta(batchvals.values.state))
    target = full_model.output_normalization_function(full_model.delta(get_targets(full_model.predict_dynamics, batchvals)))
    # break up by instances to multiply with interaction_likelihood
    if full_model.multi_instanced:
        # handle passive
        passive_error = - full_model.dist(*passive_prediction_params).log_prob(target)
        passive_error = full_model.split_instances(passive_error).sum(dim=2)
        # handle active
        pmu, pvar, ptarget = full_model.split_instances(prediction_params[0]), full_model.split_instances(prediction_params[1]), full_model.split_instances(target)
        forward_error = - full_model.dist(pmu, pvar).log_prob(ptarget).squeeze().sum(dim=2)
    else:
        forward_error = - full_model.dist(*prediction_params).log_prob(target).sum(dim=1).unsqueeze(1)
        passive_error = - full_model.dist(*passive_prediction_params).log_prob(target).sum(dim=1).unsqueeze(1)
        # forward_error = - full_model.dist(*prediction_params).log_prob(target).mean(dim=1).unsqueeze(1)
        # passive_error = - full_model.dist(*passive_prediction_params).log_prob(target).mean(dim=1).unsqueeze(1)
        # forward_error = - full_model.dist(*prediction_params).log_prob(target).max(dim=1)[0].unsqueeze(1)
        # passive_error = - full_model.dist(*passive_prediction_params).log_prob(target).max(dim=1)[0].unsqueeze(1)
        # possible to replace sum with mean, or max

    interaction_binaries, potential = full_model.compute_interaction(forward_error, passive_error, full_model.rv(target))

    if proximal is not None: 
        proximal_vector = pytorch_model.wrap(proximal[idxes], cuda=full_model.iscuda)
        interaction_binaries *= proximal_vector.unsqueeze(1)
    interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())

    run_optimizer(interaction_optimizer, full_model.interaction_model, interaction_loss)
    return batchvals, interaction_loss, interaction_likelihood, interaction_binaries

def train_combined(full_model, rollouts, test_rollout, train_args, batchvals, half_batchvals,
    trace, weights, use_weights, passive_error_all, interaction_schedule, ratio_lambda,
    active_optimizer, passive_optimizer, interaction_optimizer,
    idxes_sets=None, keep_outputs=False, proximal=None):
    inline_iter_schedule = lambda x: max(1, int(train_args.inline_iters - 1.5*train_args.inline_iters/(1+np.exp((x // train_args.num_iters)))))
    outputs = list()
    inter_loss = nn.BCELoss()
    weight_lambda = ratio_lambda
    trw = trace
    alt_weights = use_weights
    if train_args.interaction_iters > 0:
        trw = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
    for i in range(train_args.num_iters):
        # passive failure weights
        if idxes_sets is None: idxes, batchvals = rollouts.get_batch(train_args.batch_size, weights=use_weights, existing=batchvals)
        else: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, idxes=idxes_sets[i])
        # for debugging purposes only REMOVE:
        true_binaries = None
        if train_args.interaction_iters > 0 or train_args.compare_trace:
            true_binaries = trace[idxes].unsqueeze(1)
        if train_args.passive_weighting:
            pe = passive_error_all[idxes]
        # REMOVE above
        # for k, j in enumerate(idxes):
        #     print(trace[j], full_model.gamma(batchvals.values.state[k]), full_model.delta(batchvals.values.state_diff[k]))

        # print("running forward model")
        prediction_params = full_model.forward_model(full_model.gamma(batchvals.values.state))
        # print("running instance labels")
        if full_model.multi_instanced: interaction_likelihood = full_model.interaction_model.instance_labels(full_model.gamma(batchvals.values.state))
        else: interaction_likelihood = full_model.interaction_model(full_model.gamma(batchvals.values.state))
        # print("completed fmil")
        passive_prediction_params = full_model.passive_model(full_model.delta(batchvals.values.state))
        # passive_prediction_params = full_model.passive_model(full_model.delta(batchvals.values.state))
        target = full_model.output_normalization_function(full_model.delta(get_targets(full_model.predict_dynamics, batchvals)))
        # break up by instances to multiply with interaction_likelihood
        if full_model.multi_instanced:
            # handle passive
            passive_log_probs = full_model.dist(*passive_prediction_params).log_prob(target)
            passive_error = - passive_log_probs
            passive_error = full_model.split_instances(passive_error).sum(dim=2)
            # handle active
            pmu, pvar, ptarget = full_model.split_instances(prediction_params[0]), full_model.split_instances(prediction_params[1]), full_model.split_instances(target)
            forward_log_probs = full_model.dist(pmu, pvar).log_prob(ptarget).squeeze()
            forward_error = - forward_log_probs.squeeze().sum(dim=2)
        else:
            forward_log_probs = full_model.dist(*prediction_params).log_prob(target)
            forward_error = - forward_log_probs.sum(dim=1).unsqueeze(1)
            passive_log_probs = full_model.dist(*passive_prediction_params).log_prob(target)
            passive_error = - passive_log_probs.sum(dim=1).unsqueeze(1)

        # this is mislabeled, it is actually forward l1
        forward_l2 = (prediction_params[0] - target).abs().mean(dim=1).unsqueeze(1)
        # print(forward_error.shape, interaction_likelihood.shape)
        forward_loss = forward_error * interaction_likelihood.clone().detach() # detach so the interaction is trained only through discrimination 
        passive_loss = passive_error
        # might consider scaling passive error
        # print(prediction_params[0].shape, target.shape, forward_error.shape, interaction_likelihood.shape)
        active_diff = full_model.split_instances((prediction_params[0] - target)) if full_model.multi_instanced else (prediction_params[0] - target)
        if full_model.multi_instanced:
            broadcast_il = torch.stack([interaction_likelihood.clone().detach() for _ in range(active_diff.shape[-1])], dim=2)
            active_l2 = active_diff * broadcast_il
        else:
            active_l2 = (active_diff * interaction_likelihood).norm(dim=1, p=1).clone().detach()
        passive_l2 = (passive_prediction_params[0] - target).norm(dim=1, p=1)
        # append outputs
        if keep_outputs: outputs.append((prediction_params, passive_prediction_params, interaction_likelihood))

        # assign done flags
        done_flags = 1-batchvals.values.done

        # version with binarized loss
        # interaction_loss = torch.zeros((1,))
        # UNCOMMENT WHEN ACTUALLY RUNNING
        if train_args.interaction_iters <= 0:
            # redundant if interaction binaries are based on max/mean and not sum:
            forward_error = - forward_log_probs.sum(dim=1).unsqueeze(1)
            passive_error = - passive_log_probs.sum(dim=1).unsqueeze(1)
            # forward_bin_error = - forward_log_probs.mean(dim=1).unsqueeze(1)
            # passive_bin_error = - passive_log_probs.mean(dim=1).unsqueeze(1)
            # forward_bin_error = - forward_log_probs.max(dim=1)[0].unsqueeze(1)
            # passive_bin_error = - passive_log_probs.max(dim=1)[0].unsqueeze(1)

            # interaction_binaries = full_model.compute_interaction(prediction_params[0].clone().detach(), passive_prediction_params[0].clone().detach(), self.rv(target))
            interaction_binaries, potential = full_model.compute_interaction(forward_error, passive_error, full_model.rv(target))
            # print(interaction_binaries.shape, forward_error.shape, passive_error.shape)
            # interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
            # forward_bin_loss = forward_error * interaction_binaries
            # forward_max_loss = forward_error * torch.max(torch.cat([interaction_binaries, interaction_likelihood.clone(), potential.clone()], dim=1).detach(), dim = 1)[0]
            if i < train_args.passive_weight_interaction_iters: # uses high passive error as a binary for the interaction model initially, which works best with proximity
                weight_bins =  pytorch_model.wrap(weights[idxes].astype(float), cuda=full_model.iscuda).unsqueeze(1)
                interaction_final = torch.max(torch.cat([weight_bins, interaction_binaries, interaction_likelihood.clone()], dim=1).detach(), dim = 1)[0]
            else:
                interaction_final = torch.max(torch.cat([interaction_binaries, interaction_likelihood.clone()], dim=1).detach(), dim = 1)[0]
            if proximal is not None: 
                proximal_vector = pytorch_model.wrap(proximal[idxes], cuda=full_model.iscuda)
                interaction_final *=  proximal_vector
            forward_max_loss = forward_error * interaction_final
            # run_optimizer(interaction_optimizer, full_model.interaction_model, interaction_loss) # no done flags because it is possible that the interaction here is valid
        else:
            # in theory, this works even in multiinstanced settings
            interaction_binaries, potential = full_model.compute_interaction(forward_error, passive_error, full_model.rv(target))
            interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
            forward_max_loss = forward_loss
        # MIGHT require no-grad to step passive_error correctly
        # loss = (passive_error + forward_loss * interaction_schedule(i) + forward_error * (1-interaction_schedule(i))).sum() + interaction_loss.sum()
        loss = (forward_max_loss * interaction_schedule(i) + forward_error * (1-interaction_schedule(i))) * done_flags
        if train_args.intrain_passive: run_optimizer(passive_optimizer, full_model.passive_model, passive_loss)
        run_optimizer(active_optimizer, full_model.forward_model, loss)

        # PASSIVE OPTIMIZATION SEPARATE NO WEIGHTS CODE:
        # MIGHT 
        # pidxes, pbatchvals = rollouts.get_batch(train_args.batch_size, existing=pbatchvals)
        # passive_prediction_params = full_model.passive_model(full_model.delta(batchvals.values.state))
        # ptarget = full_model.network_args.output_normalization_function(full_model.delta(get_targets(full_model.predict_dynamics, pbatchvals)))
        # passive_error = - full_model.dist(*passive_prediction_params).log_prob(target)
        # PASSIVE IGNORES ACTIVE SUCCESS
        # run_optimizer(train_args, passive_optimizer, full_model.passive_model, passive_error)
        # passive_error = passive_error * (1-interaction_binaries)
        # passive_optimizer.optimizer.zero_grad()
        # (loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.full_model.passive_model.parameters(), train_args.max_grad_norm)
        # passive_optimizer.optimizer.step()            

        # INTERACTION OPTIMIZER SEPARATE
        if i % train_args.log_interval == 0:
            _combined_logging(full_model, train_args, rollouts, test_rollout, i, batchvals,
                     interaction_likelihood, interaction_binaries, true_binaries,
                     prediction_params, passive_prediction_params,
                     target, active_l2, passive_l2, done_flags, interaction_schedule,
                     forward_error, forward_loss, passive_error, trace)
            # REWEIGHTING CODE
            if train_args.interaction_iters > 0:
                weight_lambda = train_args.interaction_weight * np.exp(-i/train_args.num_iters * 5)
                print(weight_lambda)
                _, use_weights, live, dead, ratio_lambda = get_weights(ratio_lambda=weight_lambda, weights=trw, temporal_local=train_args.interaction_local)
            elif train_args.passive_weighting:
                rwlb = 5 if proximal is not None else 1 
                weight_lambda = train_args.passive_weighting * np.exp(-i/train_args.num_iters * rwlb)
                print("weight lambda", weight_lambda)
                _, use_weights, live, dead, ratio_lambda = get_weights(passive_error_all, ratio_lambda=weight_lambda, 
                    passive_error_cutoff=train_args.passive_error_cutoff, temporal_local=train_args.interaction_local, use_proximity=proximal)
                interaction_values = get_interaction_vals(full_model, rollouts)
                
                if i > 0 and train_args.interaction_boosting > 0: alt_weights = get_weights(interaction_values, ratio_lambda=weight_lambda, passive_error_cutoff=full_model.interaction_prediction)

        if train_args.interaction_iters <= 0:
            inline_iters = 1 if train_args.inline_iters == 1 else inline_iter_schedule(i)
            for ii in range(train_args.inline_iters):
                inter_batchvals, inter_l, inter_likely, inter_bins = _train_combined_interaction(full_model, train_args, rollouts, idxes_sets,
                                                                                             half_batchvals, batchvals, proximal, use_weights,
                                                                                             alt_weights, inter_loss, interaction_optimizer)
    return outputs
    # torch.save(self.forward_model, "data/active_model.pt")
    # torch.save(self.interaction_model, "data/train_int.pt")