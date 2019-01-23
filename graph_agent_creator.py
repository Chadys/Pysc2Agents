import sys
import matplotlib.pyplot as plt


def main(file_path):
    data = [[], [], []]
    formats = ['r-', 'b-', 'g-']
    legends = ['% Loss', '% Draws', '% Win']

    plt.xlabel('Episode')
    fig, ax = plt.subplots()

    with open(file_path, 'r') as f:
        plays = f.read()
        for x, reward in enumerate(plays):
            reward = int(reward)
            if x % 100 == 0:
                for d in data:
                    d.append(0)
            data[reward][-1] += 1
    for y, f, legend in zip(data, formats, legends):
        ax.plot(range(100, (len(y)*100)+100, 100), y, f, legend)
    plt.plot(data[0], 'r-', data[1], 'b-', data[2], 'g-')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])