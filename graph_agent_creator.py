import sys
import matplotlib.pyplot as plt


def main(file_path):
    data = [[], [], []]
    formats = ['r-', 'b-', 'g-']
    legends = ['% Loss', '% Draws', '% Win']

    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Percentage')

    with open(file_path, 'r') as f:
        plays = f.read()
        # get number of parties, rounded by 100
        n_plays = len(plays) // 100 * 100
        for x, reward in enumerate(plays):
            if x >= n_plays:
                break
            reward = int(reward)
            if x % 100 == 0:
                for d in data:
                    d.append(0)
            data[reward][-1] += 1
    for y, f, legend in zip(data, formats, legends):
        ax.plot(range(100, (len(y)*100)+100, 100), y, f, label=legend)

    ax.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])