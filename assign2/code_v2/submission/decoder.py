import argparse
from collections import deque

SUITS = {'H': 0, 'D': 1}
SUIT_NAMES = {0: 'H', 1: 'D'}
ALL_CARDS = tuple(sorted([(v, s) for v in range(1, 14) for s in SUITS.values()]))

def generate_states(threshold):
    state_map = {(): 0}  # Maps hand tuple -> state index
    index_map = [()]    # Maps state index -> hand tuple
    queue = deque([()])
    while queue:
        curr_hand = queue.popleft()
        curr_hand_set = set(curr_hand)
        for card in ALL_CARDS:
            if card not in curr_hand_set:
                new_hand_list = sorted(list(curr_hand) + [card])
                new_hand_tuple = tuple(new_hand_list)
                total = sum(c[0] for c in new_hand_tuple)
                if(total<threshold and (new_hand_tuple not in state_map)):
                    new_index = len(state_map)
                    state_map[new_hand_tuple] = new_index
                    index_map.append(new_hand_tuple)
                    queue.append(new_hand_tuple)
    return state_map, index_map

def parse_card(s):
    suit = s[-1]
    value = int(s[:-1])
    return (value, SUITS[suit])

def card_tuple(card):
    value,suit = card
    return f"{value}{SUIT_NAMES[suit]}"

def sort_key(hand):
    return (len(hand),) + hand

def main():
    parser = argparse.ArgumentParser(description="Decode MDP policy for card game.")
    parser.add_argument("--value_policy", required=True, help="Path to the value and policy file from planner.py.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--testcase", help="Path to the testcase file with hand instances.")
    group.add_argument("--automate", help="Path to a game config file to generate a full policy.")
    args = parser.parse_args()
    with open(args.value_policy, 'r') as f:
        policy = [int(line.strip().split()[1]) for line in f]
    if args.automate:
        with open(args.automate, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            threshold = int(lines[1])
        _, index_map = generate_states(threshold)
        output_data = []
        for i,hand in enumerate(index_map):
            if not hand: continue            
            action = policy[i]
            hand_str = " ".join(map(card_tuple, hand))
            output_data.append((hand, f"{hand_str} -> {action}"))
        output_data.sort(key=lambda item: sort_key(item[0]))
        for _, line in output_data:
            print(line)
    else:
        with open(args.testcase, 'r') as f:
            lines = f.readlines()
            threshold = int(lines[1].strip())
            hand_lines = lines[5:]
        state_map, _ = generate_states(threshold)
        for line in hand_lines:
            hand_lst = line.strip().split()
            if not hand_lst:
                print(0) # if empty hand, action is draw
                continue
            hand_tuple = tuple(sorted([parse_card(s) for s in hand_lst]))
            state = state_map[hand_tuple]
            action = policy[state]
            print(action)

if __name__ == "__main__":
    main()