import argparse
import itertools
import numpy as np

from args import ArgsConfig
from plot import PlotLinesHandler


def truncated_normal() -> float:
    rnt = np.random.normal(loc=0.5, scale=1)
    while rnt > 1.0 or rnt < 0.0:
        rnt = np.random.normal(loc=0.5, scale=1)
    return rnt


def draw(p) -> bool:
    return True if np.random.uniform() < p else False


class Consultant:
    _ids = itertools.count(0)

    def __init__(self, inno_init) -> None:
        self.id = next(self._ids)
        
        self.inno = inno_init
        self.quality = truncated_normal()
        self.r_last = 0
        self.r_cur = 0
        self.a_h_last = 0

        self.clients = []
        self.o_avg_list = []


    def receive_return(self, val_return: float):
        self.r_cur += val_return


class Firm:
    _ids = itertools.count(0)

    def __init__(self, inno_init) -> None:
        self.id = next(self._ids)

        self.inno = inno_init
        self.o_last = 0
        self.o_cur = 0
        self.a_h_last = 0

        self.consultant = None
    

    def receive_outcome(self, val_outcome: float):
        self.o_cur += val_outcome


class Market:

    def __init__(self, args: argparse.ArgumentParser) -> None:
        np.random.seed(args.rnd_seed)
        self.args = args

        # consultant
        self.cons = [Consultant(inno) for inno in np.random.randint(low=0, high=self.args.n_innovation, size=self.args.n_consultant)]
        self.inno_to_con = [{"con": [], "prob_w": []} for _ in range(self.args.n_innovation)]
        for con in self.cons:
            self.inno_to_con[con.inno]["con"].append(con)
            self.inno_to_con[con.inno]["prob_w"].append(0 + self.args.c)
        
        # firm
        self.firms = [Firm(inno) for inno in np.random.choice(self.get_inno_pool(), size=self.args.n_firm)]
        self.inno_to_firm_n = [0 for _ in range(self.args.n_innovation)]
        for firm in self.firms:
            self.inno_to_firm_n[firm.inno] += 1
        
        # innovation
        self.inno_V = [truncated_normal() for _ in range(self.args.n_innovation)]

        # record
        self.firm_adp_rate = list()
        self.firm_most_inno = list()
        self.con_adp_rate = list()
        self.con_most_inno = list()

        # initial matches
        for firm in self.firms:
            selected_con = self.select_con_from_inno(inno_id=firm.inno)
            firm.consultant = selected_con
            selected_con.clients.append(firm)
        
    
    def select_con_from_inno(self, inno_id) -> Consultant:
        prob = np.array(self.inno_to_con[inno_id]["prob_w"])
        prob = prob / np.sum(prob)
        prob[np.argmax(prob)] += 1 - np.sum(prob) # adjust it to make its sum be exactly 1.0
        selected_con_id = np.random.choice(np.arange(len(self.inno_to_con[inno_id]["con"])), p=prob)
        return self.inno_to_con[inno_id]["con"][selected_con_id]
    

    def get_inno_pool(self) -> list:
        """ Return a list of all innovation that at least one consultant supplies. """
        return np.array([inno for inno in range(self.args.n_innovation) if self.inno_to_con[inno]["con"]])
        

    def simulate_step(self) -> None:
        # 1: consultants have an innovation to supply
        # 2: firms have an innovation to demand
        # 3: a firm select a consultant
        
        # 5: the firm recieves an outcome
        for firm in self.firms:
            firm_outcome = self.args.alpha * self.inno_V[firm.inno] + \
                           self.args.beta * firm.consultant.quality + \
                           (1-self.args.alpha-self.args.beta) * truncated_normal()
            firm.receive_outcome(firm_outcome)
            
        # 4: the consultant recieves a return
        for con in self.cons:
            con_return = self.args.eta * len(con.clients) * \
                         (self.inno_to_firm_n[con.inno]/len(self.inno_to_con[con.inno]["con"]))
            con.receive_return(con_return)
            if con.clients:
                con.o_avg_list.append([firm.o_cur for firm in con.clients])
                if len(con.o_avg_list) > self.args.window:
                    del con.o_avg_list[0]
                assert len(con.o_avg_list) <= self.args.window
        
        # 6: consultants decide whether to make a change
        r_cur_arr = np.array([con.r_cur for con in self.cons])
        best_con_idx = np.argmax(r_cur_arr)
        best_con_inno = self.cons[best_con_idx].inno
        sum_r_cur = np.sum(r_cur_arr)

        for con in self.cons:
            aspi_history = (1-self.args.xi_c) * con.a_h_last + self.args.xi_c * con.r_last
            aspi_social = (sum_r_cur - con.r_cur) / (self.args.n_consultant - 1)
            aspi_total = self.args.gamma_c * aspi_history + (1-self.args.gamma_c) * aspi_social
            prob_change = 1 / (1 + np.exp(self.args.a_c + self.args.b_c*(con.r_cur - aspi_total)))

            # find the index of con in self.inno_to_con[con.inno]["con"]
            con_index = 0
            while self.inno_to_con[con.inno]["con"][con_index].id != con.id:
                con_index += 1
                assert con_index < len(self.inno_to_con[con.inno]["con"])

            if draw(prob_change):
                con_ori_inno = con.inno
                if draw(self.args.p_mimic_c):
                    con.inno = best_con_inno
                else:
                    con.inno = np.random.choice(np.arange(self.args.n_innovation))
                
                if con_ori_inno != con.inno:
                    # remove from the original inno
                    self.inno_to_con[con_ori_inno]["con"].pop(con_index)
                    self.inno_to_con[con_ori_inno]["prob_w"].pop(con_index)

                    # add to new inno
                    self.inno_to_con[con.inno]["con"].append(con)
                    self.inno_to_con[con.inno]["prob_w"].append(0 + self.args.c)

                    con.o_avg_list = []

                    # its clients reselect
                    if self.inno_to_con[con_ori_inno]["con"]:
                        for firm in con.clients:
                            firm.consultant = self.select_con_from_inno(inno_id=con_ori_inno)
                            firm.consultant.clients.append(firm)
                    else:
                        for firm in con.clients:
                            self.inno_to_firm_n[firm.inno] -= 1
                            firm.inno = np.random.choice(self.get_inno_pool())
                            firm.consultant = self.select_con_from_inno(inno_id=firm.inno)
                            firm.consultant.clients.append(firm)
                            self.inno_to_firm_n[firm.inno] += 1
                    
                    con.clients = []
            
            else:
                all_o = []
                for l in con.o_avg_list:
                    all_o += l
                o_avg = sum(all_o) / len(all_o) if all_o else 0
                self.inno_to_con[con.inno]["prob_w"][con_index] = o_avg + self.args.c
            
            # update consultant data
            con.a_h_last = aspi_history
            con.r_last = con.r_cur
            con.r_cur = 0

        # 7: firms decide whether to make a change
        o_cur_arr = np.array([firm.o_cur for firm in self.firms])
        best_inno_sort = [self.firms[firm_idx].inno for firm_idx in np.argsort(o_cur_arr)[::-1]]
        tmp_idx = 0
        while len(self.inno_to_con[best_inno_sort[tmp_idx]]["con"]) == 0:
            tmp_idx += 1
        best_firm_inno = best_inno_sort[tmp_idx]

        sum_o_cur = np.sum(o_cur_arr)
        inno_pool_con = self.get_inno_pool()

        for firm in self.firms:
            aspi_history = (1-self.args.xi_f) * firm.a_h_last + self.args.xi_f * firm.o_last
            aspi_social = (sum_o_cur - firm.o_cur) / (self.args.n_firm - 1)
            aspi_total = self.args.gamma_f * aspi_history + (1-self.args.gamma_f) * aspi_social
            prob_change = 1 / (1 + np.exp(self.args.a_f + self.args.b_f*(firm.o_cur - aspi_total)))

            if draw(prob_change):
                firm_ori_inno = firm.inno
                if draw(self.args.p_mimic_f):
                    firm.inno = best_firm_inno
                else:
                    firm.inno = np.random.choice(inno_pool_con)
                
                if firm_ori_inno != firm.inno:
                    # consultant lose a client
                    client_index = 0
                    while firm.consultant.clients[client_index].id != firm.id:
                        client_index += 1
                    firm.consultant.clients.pop(client_index)

                    # firm select a new consultant supplying firm.inno
                    firm.consultant = self.select_con_from_inno(inno_id=firm.inno)
                    firm.consultant.clients.append(firm)

                    self.inno_to_firm_n[firm_ori_inno] -= 1
                    self.inno_to_firm_n[firm.inno] += 1
            
            # update firm data
            firm.a_h_last = aspi_history
            firm.o_last = firm.o_cur
            firm.o_cur = 0
    
    
    def simulate(self):
        for step in range(self.args.n_periods):
            self.simulate_step()

            firm_len = np.array(self.inno_to_firm_n)
            assert np.sum(firm_len) == self.args.n_innovation
            self.firm_adp_rate.append(np.max(firm_len)/self.args.n_innovation)
            self.firm_most_inno.append(np.argmax(firm_len))

            con_len = np.array([len(d["con"]) for d in self.inno_to_con])
            assert np.sum(con_len) == self.args.n_innovation
            self.con_adp_rate.append(np.max(con_len)/self.args.n_innovation)
            self.con_most_inno.append(np.argmax(con_len))

            print("step {:3d} | firm: {:.4f}; most_inno: {:2d} | con: {:.4f}; most_inno: {:2d}".format(step,
                self.firm_adp_rate[-1], self.firm_most_inno[-1], self.con_adp_rate[-1], self.con_most_inno[-1]))




if __name__ == "__main__":
    args_config = ArgsConfig()
    args = args_config.get_args()
    plh_firm = PlotLinesHandler(xlabel="Iterations", ylabel="Adoption", ylabel_show="% Adoption",
                                x_lim=args.n_periods+15, y_lim=95)
    plh_con = PlotLinesHandler(xlabel="Iterations", ylabel="Adoption", ylabel_show="% Adoption",
                               x_lim=args.n_periods+15, y_lim=95)

    m = Market(args)
    m.simulate()

    plh_firm.plot_line(np.array(m.firm_adp_rate), color="black")
    plh_firm.plot_changes(m.firm_most_inno, m.firm_adp_rate, color="black")
    plh_firm.save_fig(title_param="rndSeed_{}_firm".format(args.rnd_seed))

    plh_con.plot_line(np.array(m.con_adp_rate), color="black")
    plh_con.plot_changes(m.con_most_inno, m.con_adp_rate, color="black")
    plh_con.save_fig(title_param="rndSeed_{}_con".format(args.rnd_seed))

