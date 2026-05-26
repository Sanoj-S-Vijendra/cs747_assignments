#!/usr/bin/env python3

import argparse
from collections import deque

SUITS = {'H': 0, 'D': 1}
SUIT_NAMES = {0: 'H', 1: 'D'}
ALL_CARDS = tuple(sorted([(v, s) for v in range(1, 14) for s in SUITS.values()]))

def card_to_action(card):
    value,suit = card
    if suit == SUITS['H']: # H: 1-13
        return value
    else:
        return value+13 # D: 14-26

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

def check_sequence(hand, sequence):
    if not sequence:
        return False
    hand_values = sorted(list(set([card[0] for card in hand])))
    seq_len = len(sequence)
    for i in range(len(hand_values) - seq_len + 1):
        if tuple(hand_values[i:i+seq_len]) == sequence:
            return True
    return False

def print_mdp(s_map, i_map, thr, bonus, seq):
    num_hand_states = len(s_map)
    S_stop = num_hand_states
    S_bust = num_hand_states+1
    print(f"numStates {num_hand_states+2}")
    print(f"numActions 28")
    print(f"end {S_stop} {S_bust}")
    all_cards_set = set(ALL_CARDS)
    for s1,hand1 in enumerate(i_map):
        hand1_set = set(hand1)
        remaining_deck = all_cards_set-hand1_set
        if remaining_deck:
            prob_draw = 1.0/len(remaining_deck)
            bust_prob = 0.0
            for card_drawn in remaining_deck:
                new_hand = tuple(sorted(list(hand1)+[card_drawn]))
                if sum(c[0] for c in new_hand)>=thr:
                    bust_prob+=prob_draw
                else:
                    s2 = s_map[new_hand]
                    print(f"transition {s1} 0 {s2} 0.0 {prob_draw:.10f}")
            if bust_prob > 0:
                print(f"transition {s1} 0 {S_bust} 0.0 {bust_prob:.10f}")
        if hand1: # swap only for non empty hands
            for card in hand1:
                action = card_to_action(card)
                bust_prob = 0.0
                if remaining_deck:
                    prob_draw = 1.0/len(remaining_deck)
                    for card_drawn in remaining_deck:
                        temp_hand = list(hand1)
                        temp_hand.remove(card)
                        new_hand = tuple(sorted(temp_hand + [card_drawn]))
                        if sum(c[0] for c in new_hand)>=thr:
                            bust_prob += prob_draw
                        else:
                            s2 = s_map[new_hand]
                            print(f"transition {s1} {action} {s2} 0.0 {prob_draw:.10f}")
                    if bust_prob > 0:
                        print(f"transition {s1} {action} {S_bust} 0.0 {bust_prob:.10f}")
        total = sum(c[0] for c in hand1)
        final_score = total
        if check_sequence(hand1, seq):
            final_score+=bonus
        print(f"transition {s1} 27 {S_stop} {final_score} 1.0")
    print("mdptype episodic")
    print("discount 1.0")

def main():
    parser = argparse.ArgumentParser(description="Encode a card game into an MDP.")
    parser.add_argument("--game_config", required=True, help="Path to the game specification file.")
    args = parser.parse_args()
    with open(args.game_config, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        threshold = int(lines[1])
        bonus = int(lines[2])
        sequence = tuple(map(int, lines[3].split()))
    state_map, index_map = generate_states(threshold)
    print_mdp(state_map, index_map, threshold, bonus, sequence)

if __name__ == "__main__":
    main()