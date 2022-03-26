import argparse
import itertools
import numpy as np

from args import ArgsConfig


def truncated_normal() -> float:
    return max(min(np.random.normal(loc=0.5, scale=1), 1.0), 0.0)

def draw(p) -> bool:
    return True if np.random.uniform() < p else False


class Consultant:
    _ids = itertools.count(0)

    def __init__(self, inno_init) -> None:
        self.id = next(self._ids)
        
        self.inno = inno_init
        self.quality = truncated_normal()
        self.inno_to_con_index = None

        self.r_last = 0
        self.r_cur = 0
        self.a_h_last = 0

        self.o_avg_list = [list()]


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
    

    def receive_outcome(self, val_outcome: float):
        self.o_cur += val_outcome


class Market:

    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.args = args
        
        # firm
        self.firms = [Firm(inno) for inno in np.random.randint(low=0, high=self.args.n_firm, size=self.args.n_firm)]
        self.inno_to_firm = [[] for _ in range(self.args.n_innovations)]
        for firm in self.firms:
            self.inno_to_firm[firm.inno].append(firm)
        
        # consultant
        self.cons = [Consultant(inno) for inno in np.random.randint(low=0, high=self.args.n_consultant, size=self.args.n_consultant)]
        self.inno_to_con = list()
        for _ in range(self.args.n_innovation):
            d = dict()
            d["con"] = list()
            d["prob_w"] = list()
            self.inno_to_con.append(d)
        for con in self.cons:
            con.inno_to_con_index = len(self.inno_to_con[con.inno]["con"])
            self.inno_to_con[con.inno]["con"].append(con)
            self.inno_to_con[con.inno]["prob_w"].append(0 + self.args.c)
        
        # innovation
        self.inno_V = [truncated_normal() for _ in range(self.args.n_innovation)]

        # record
        self.firm_adp_rate = list()
        self.con_adp_rate = list()
        
    
    def select_con_from_inno(self, inno_id) -> Consultant:
        prob = np.array(self.inno_to_con[inno_id]["prob_w"])
        prob = prob / np.sum(prob)
        prob[np.argmax(prob)] += 1 - np.sum(prob) # adjust it to make its sum be exactly 1.0
        selected_con_id = np.random.choice(np.arange(len(self.inno_to_con[inno_id]["con"]), p=prob))
        return self.inno_to_con[inno_id]["con"][selected_con_id]
        

    def simulate_step(self):
        # 1: consultants have an innovation to supply
        # 2: firms have an innovation to demand
        
        for firm in self.firms:
            # 3: a firm select a consultant
            selected_con = self.select_con_from_inno(firm, inno_id=firm.inno)
            
            # 4: the firm recieves an outcome
            firm_outcome = self.args.alpha * self.inno_V[firm.inno] + \
                           self.args.beta * selected_con.quality + \
                           (1-self.args.alpha-self.args.beta) * truncated_normal()
            firm.receive_outcome(firm_outcome)
            
            # 5: the consultant recieves a return
            con_return = self.eta * (len(self.inno_to_firm[firm.inno])/len(self.inno_to_con[firm.inno]["con"]))
            selected_con.receive_return(con_return)
            selected_con.o_avg_list[-1].append(firm_outcome)
        
        # 6: consultants decide whether to make a change
        r_cur_arr = np.array([con.r_cur for con in self.cons])
        best_con_idx = np.argmax(r_cur_arr)
        best_con_inno = self.cons[best_con_idx].inno
        sum_r_cur = np.sum(r_cur_arr)

        for con in self.cons:
            aspi_history = (1-self.args.xi_c) * firm.a_h_last + self.args.xi_c * con.r_last
            aspi_social = (sum_r_cur - con.r_cur) / (self.n_consultant - 1)
            aspi_total = self.args.gamma_c * aspi_history + (1-self.args.gamma_c) * aspi_social
            prob_change = 1 / (1 + np.exp(self.args.a_c + self.args.b_c*(con.r_cur - aspi_total)))

            if draw(prob_change):
                # remove from the original inno
                con_index = 0
                while self.inno_to_con[con.inno]["con"][con_index].id != con.id:
                    con_index += 1
                    assert con_index < len(self.inno_to_con[con.inno]["con"])
                self.inno_to_con[con.inno]["con"].pop(con_index)
                self.inno_to_con[con.inno]["prob_w"].pop(con_index)

                if draw(self.args.p_mimic_f):
                    con.inno = best_con_inno
                else:
                    con.inno = np.random.choice(np.arange(self.args.n_innovation))
                
                # add to new inno
                con.inno_to_con_index = len(self.inno_to_con[con.inno]["con"])
                self.inno_to_con[con.inno]["con"].append(con)
                self.inno_to_con[con.inno]["prob_w"].append(0 + self.args.c)

                con.o_avg_list = [list()]
            
            else:
                # update o_avg
                if len(con.o_avg_list) > self.args.window:
                    del con.o_avg_list[0]
                all_o = []
                for l in con.o_avg_list:
                    all_o += l
                o_avg = sum(all_o) / len(all_o)
                self.inno_to_con[con.inno]["prob_w"][con.inno_to_con_index] = o_avg + self.args.c
            
            # update firm data
            con.a_h_last = aspi_history
            con.r_last = con.r_cur
            con.r_cur = 0

        # 7: firms decide whether to make a change
        o_cur_arr = np.array([firm.o_cur for firm in self.firms])
        best_firm_idx = np.argmax(o_cur_arr)
        best_firm_inno = self.firms[best_firm_idx].inno
        sum_o_cur = np.sum(o_cur_arr)
        inno_pool_con = np.array([inno for inno in range(self.args.n_innovation) if self.inno_to_con[inno]["con"]])

        for firm in self.firms:
            aspi_history = (1-self.args.xi_f) * firm.a_h_last + self.args.xi_f * firm.o_last
            aspi_social = (sum_o_cur - firm.o_cur) / (self.n_firm - 1)
            aspi_total = self.args.gamma_f * aspi_history + (1-self.args.gamma_f) * aspi_social
            prob_change = 1 / (1 + np.exp(self.args.a_f + self.args.b_f*(firm.o_cur - aspi_total)))

            if draw(prob_change):
                # remove from the original inno
                firm_index = 0
                while self.inno_to_firm[firm_index][firm.inno].id != firm.id:
                    firm_index += 1
                    assert firm_index < len(self.inno_to_firm[firm.inno])
                self.inno_to_firm[firm.inno].pop(firm_index)

                if draw(self.args.p_mimic_f):
                    firm.inno = best_firm_inno
                else:
                    firm.inno = np.random.choice(inno_pool_con)
                
                # add to new inno
                self.inno_to_firm[firm.inno].append(firm)
            
            # update firm data
            firm.a_h_last = aspi_history
            firm.o_last = firm.o_cur
            firm.o_cur = 0
    
    
    def simulate(self):
        for step in range(self.args.n_periods):
            self.simulate_step()

            firm_len = np.array([len(inno_list) for inno_list in self.inno_to_firm])
            assert np.sum(firm_len) == self.args.n_innovation
            self.firm_adp_rate.append(np.max(firm_len)/self.args.n_innovation)

            con_len = np.array([len(d["con"]) for d in self.inno_to_con])
            assert np.sum(con_len) == self.args.n_innovation
            self.con_adp_rate.append(np.max(con_len)/self.args.n_innovation)

            print("step {:3d} | firm: {:.4f}; con: {:.4f}".format(step,
                self.firm_adp_rate[-1], self.con_adp_rate[-1]))




if __name__ == "__main__":
    args_config = ArgsConfig()
    args = args_config.get_args()
    m = Market(args)
    m.simulate()
