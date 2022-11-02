from os import path as osp
import numpy as np
import pickle
import random

import torch
import torch.nn as nn

from model.plot import Plot
import sifid.sifid

from imaginaire.losses.gan import GANLoss

class Loss():
    def __init__(self, args, num_classes):
        self.args = args
        self.num_classes = num_classes
        self.model_checkpoints_folder = osp.join("train", self.args.exp)
        self.plot = Plot(args, num_classes)
        self.gan_loss = GANLoss(self.args.gan_loss_method)
        self.mode = ""
        self.logs = self.load_logs()

    def set_mode(self, value):
        self.mode = value

    # we initially call this function to get the log dict
    # this log dict either is initially empty or, if we continue training loaded from a previous stage
    # if we load a previous stage, we might have added a new sub-dict, that was not part of the training,
    # which is than added to not run into an error. However, we print which sub-dicts have been added to the object
    def load_logs(self):
        init_log = self.load_init_logging_file()
        if self.args.load_checkpoint:
            try:
                logs_old = pickle.load(open(osp.join(self.model_checkpoints_folder, "objects", "logs.obj"), 'rb'))
                if logs_old.keys() == init_log.keys():
                    print("loaded logs successfully")
                    return logs_old
                else:
                    missing_keys = list(set(init_log) - set(logs_old))
                    for key in missing_keys:
                        logs_old[key] = init_log[key]
                    print("Added the following keys to the log dict, because they were missing: " + ''.join(missing_keys))
                    return logs_old
            except:
                raise("Not able to load log data, use init dict")
        else:
            return init_log

    # initialize the loss dict, used to log loss values, logits and plot their values together with a headline
    def load_init_logging_file(self):
        return dict(store=dict(recon_loss=[],
                               fm_loss=[],
                               gan_gen_loss=[],
                               gan_dis_loss=[],
                               cur_loss=[],
                               dis_win_rate=[]),
                    logits=dict(dis=dict(zip(range(self.num_classes),
                                             [dict(fake=[], true=[]) for _ in range(self.num_classes)])),
                                gen=dict(zip(range(self.num_classes),
                                             [dict(fake=[]) for _ in range(self.num_classes)]))),
                    sifid_scores=[],
                    headline=dict(recon_loss="Pixel Reconstruction Loss",
                                  fm_loss="Feature Matching Loss",
                                  gan_gen_loss="GAN Loss within Generator",
                                  gan_dis_loss="GAN Loss within Discriminator",
                                  cur_loss="Global Loss",
                                  dis_win_rate="Winning rate of the discriminator"))

    # save the log dict to experiment dir
    def save_logs(self):
        try: pickle.dump(self.logs, open(osp.join(self.model_checkpoints_folder, "objects", "logs.obj"), 'wb'))
        except: raise("Error while saving log dict")

    # add some value to the log dict at some key
    def _write_log(self, key, value):
        self.logs["store"][key].append(np.float(value.detach()))

    # functions that calls the write function to add a value to some key in our log dict
    # we split both functions because we dont calculate a win rate during validation and could therefore not log this
    # value during training
    def log_loss(self, log, value):
        if self.mode == "val":
            self._write_log(log, value)

    # add a logit to our log dict
    # we dont use the same function as we would for losses because we store the logits of different labels in different
    # dicts, such that we need a second log function
    # also we do some processing on those logits because we effectively use a Markov random field and get multiple
    # logits each image
    def log_logit(self, label, mode, key, logit):
        logit = np.float(logit.mean(3).mean(2).squeeze(1).detach())
        self.logs["logits"][mode][label][key].append(logit)

    # since we have a batch with multiple images, which have multiple labels and we store the logits of
    # different labels distinct from each other, we need to process each logit individually
    # while being in validation mode
    def log_logits(self, label, mode, key, logits):
        if self.mode == "val":
            if logits.size()[0] == 1:
                self.log_logit(label[0], mode, key, logits)
            else:
                for i, logit in enumerate(logits):
                    self.log_logit(label[i], mode, key, logit.unsqueeze(0))

    # we calculate a sifid score over the samples, which we saved during the current validation
    # sifid uses the metric of fid to rate the image quality, but uses the feature map right before the
    # second pooling layer of the Inception v3 model by Szegedy et al.
    # afterwards we log those scores
    def get_sifid_score(self):
        path = osp.join(self.model_checkpoints_folder, "plots", "sifid_images")
        score = sifid.sifid.calculate_sifid_given_paths(osp.join(path, "true"), osp.join(path, "false"),
                                                        1, self.args.d_id, 64, "jpg")
        self.logs["sifid_scores"].append(np.mean(score))

    # based on Arjovsky and Bottou low dimensional manifolds are almost certainly disjoint
    # we therefore inject noise to D, to make true and false probabilities distribution overlap
    # we set true = false and false = true, to make this happen
    def rnd_flip_label(self, true_out, fake_out):
        if random.random() <= self.args.D_noise:
            temp = true_out
            true_out = fake_out
            fake_out = temp

        return true_out, fake_out

    # we use the hinge loss to get the GAN loss for D
    # also we use a might flip labels by a given propability
    # we multiply with the respective lambda value to weight its loss
    # afterwards we log its result
    def gan_loss_D(self):
        true_out = self.dis_obj['true_disc_out']
        fake_out = self.dis_obj['fake_disc_out']
        if self.args.D_noise > 0: true_out, fake_out = self.rnd_flip_label(true_out, fake_out)

        fake_loss = self.gan_loss(fake_out, False, dis_update=True)
        true_loss = self.gan_loss(true_out, True, dis_update=True)
        loss = (true_loss + fake_loss) * self.args.lg

        self.log_loss("gan_dis_loss", loss)
        self.log_logits(self.gen_obj['style_label_1'].cpu().numpy(), "dis", "true", self.dis_obj['true_disc_out'])
        self.log_logits(self.gen_obj['style_label_1'].cpu().numpy(), "dis", "fake", self.dis_obj['fake_disc_out'])
        return loss

    # we use the hinge loss to get the GAN loss for G, which is the non-saturated loss for G
    # we multiply with the respective lambda value to weight its loss
    # afterwards we log its result
    def gan_loss_G(self):
        loss = self.gan_loss(self.dis_obj['fake_disc_out'], True, dis_update=False)
        loss *= self.args.lg
        self.log_loss("gan_gen_loss", loss)
        self.log_logits(self.gen_obj['style_label_1'].cpu().numpy(), "gen", "fake", self.dis_obj['fake_disc_out'])
        return loss

    # calculate the L1 loss between the content image and G(content_image, content_image), which means
    # to stylize the image with itself
    # we multiply with the respective lambda value to weight its loss
    # afterwards we log its result
    def recon_loss(self):
        loss = nn.L1Loss()(self.gen_obj['content_image'], self.gen_obj['recon_image'])
        loss *= self.args.lrp
        self.log_loss("recon_loss", loss)
        return loss

    # feature matching loss calculates the L1 distance between the features in D beween those coming from a real
    # sample and those coming from the generated sample
    # we multiply with the respective lambda value to weight its loss
    # afterwards we log its result
    def feature_matching_loss(self):
        loss = nn.L1Loss()(self.dis_obj['fake_disc_feat'], self.dis_obj['true_disc_feat'])
        loss *= self.args.lfm
        self.log_loss("fm_loss", loss)
        return loss

    # we calulate the accuracy by considering >0: true and <0: fake
    # we get a list of true/false values, convert it to float suhc that true = 1, false = 0 and calculate a mean over it
    def calc_true_accuracy(self, output):
        return torch.mean(torch.greater(output, torch.zeros_like(output)).float())

    # we calulate the accuracy by considering >0: true and <0: fake
    # we get a list of true/false values, convert it to float such that true = 1, false = 0 and calculate a mean over it
    def calc_fake_accuracy(self, output):
        return torch.mean(torch.less(output, torch.zeros_like(output)).float())

    # calculate the accuracy for G, by using the output of D, while feeding the sample of G to D
    # we add an epsilon value to this for stability reasons
    # also done by Kotovenko et al.
    def calc_accuracy_G(self):
        return self.calc_true_accuracy(self.dis_obj['fake_disc_out']) + 1e-6

    # calculate the accuracy for D, by using the output of D, while feeding the sample of G and real samples to D
    def calc_accuracy_D(self):
        return self.calc_fake_accuracy(self.dis_obj['fake_disc_out']) * 0.5 + \
               self.calc_true_accuracy(self.dis_obj['true_disc_out']) * 0.5

    # here we calculate all losses, which we need to update G
    def compute_G_loss(self, gen_obj, dis_obj):
        self.gen_obj = gen_obj
        self.dis_obj = dis_obj
        loss = self.recon_loss()
        loss += self.feature_matching_loss()
        loss += self.gan_loss_G()
        accuracy = self.calc_accuracy_G()
        return loss, accuracy

    # here we calculate all losses, which we need to update D
    def compute_D_loss(self, gen_obj, dis_obj):
        self.gen_obj = gen_obj
        self.dis_obj = dis_obj
        loss = self.gan_loss_D()
        accuracy = self.calc_accuracy_D()
        return loss, accuracy

    # call all functions that we need to create plots for the current state
    def get_plots(self):
        self.get_sifid_score()
        self.save_logs()
        self.plot.generate_plots(self.logs)