import numpy as np
import plot
import twocarts


def main():

    day_of_the_month_youngest_group_member_birthday = 420
    day_of_the_month_oldest_group_member_birthday = 69
    group_number = 19

    input = twocarts.Input(m1=10,
                           m2=group_number,
                           k1=day_of_the_month_youngest_group_member_birthday,
                           k2=3,
                           k3=group_number/2,
                           c1=0.2,
                           c2=0.2,
                           c3=0.2,
                           F1_abs=day_of_the_month_oldest_group_member_birthday,
                           F2_abs=10,
                           F1_phase=0,
                           F2_phase=0,
                           )

    frequencies = np.linspace(1, 50)
    for frequency in frequencies:
        x, y = twocarts.twocards(**input.__dict__, omega=frequency)
        plot.plot(x, y, "frequency" + str(frequency))


if __name__ == "__main__":
    main()
