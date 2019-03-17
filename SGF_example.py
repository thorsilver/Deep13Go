from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Point
from dlgo.utils import print_board


# Honinbo Shusaku (ear-reddening move)
sgf_content = "(;B[qd];W[dc];B[pq];W[oc];B[cp];W[cf];B[ep];W[qo];B[pe];W[np]" \
              ";B[po];W[pp];B[op];W[qp];B[oq];W[oo];B[pn];W[qq];B[nq];W[on]" \
              ";B[pm];W[om];B[pl];W[mp];B[mq];W[ol];B[pk];W[lq];B[lr];W[kr]" \
              ";B[lp];W[kq];B[qr];W[rr];B[rs];W[mr];B[nr];W[pr];B[ps];W[qs]" \
              ";B[no];W[mo];B[qr];W[rm];B[rl];W[qs];B[lo];W[mn];B[qr];W[qm]" \
              ";B[or];W[ql];B[qj];W[rj];B[ri];W[rk];B[ln];W[mm];B[qi];W[rq]" \
              ";B[jn];W[ls];B[ns];W[gq];B[go];W[ck];B[kc];W[ic];B[pc];W[nj]" \
              ";B[ke];W[og];B[oh];W[pb];B[qb];W[ng];B[mi];W[mj];B[nd];W[ph]" \
              ";B[qg];W[pg];B[hq];W[hr];B[ir];W[iq];B[hp];W[jr];B[fc];W[lc]" \
              ";B[ld];W[mc];B[lb];W[mb];B[md];W[qf];B[pf];W[qh];B[rg];W[rh]" \
              ";B[sh];W[rf];B[sg];W[pj];B[pi];W[oi];B[oj];W[ni];B[qk];W[ok]" \
              ";B[qe];W[kb];B[jb];W[ka];B[jc];W[ob];B[ja];W[la];B[db];W[cc]" \
              ";B[fe];W[cn];B[gr];W[is];B[fq];W[io];B[ji];W[eb];B[fb];W[eg]" \
              ";B[dj];W[dk];B[ej];W[cj];B[dh];W[ij];B[hm];W[gj];B[eh];W[fl]" \
              ";B[fg];W[er];B[dm];W[fn];B[dn];W[gn];B[jj];W[jk];B[kk];W[ii]" \
              ";B[ik];W[jl];B[kl];W[il];B[jh];W[co];B[do];W[ih];B[hn];W[hl]" \
              ";B[bl];W[dg];B[gh];W[ch];B[ig];W[ec];B[cr];W[fd];B[gd];W[ed]" \
              ";B[gc];W[bk];B[cm];W[gs];B[gp];W[li];B[kg];W[in];B[lj];W[lg]" \
              ";B[gm];W[jf];B[jg];W[im];B[fm];W[kf];B[lf];W[mf];B[le];W[gf]" \
              ";B[hf];W[ff];B[gg];W[lk];B[kj];W[km];B[lm];W[ll];B[jm];W[ge]" \
              ";B[he];W[ef];B[ea];W[cb];B[fr];W[fs];B[dr];W[qa];B[ra];W[pa]" \
              ";B[rb];W[da];B[gi];W[fj];B[fi];W[fa];B[ga];W[gl];B[ek];W[em]" \
              ";B[ho];W[el];B[en];W[jo];B[kn];W[ci];B[lh];W[mh];B[mg];W[di]" \
              ";B[ei];W[lg];B[qn];W[rn];B[re];W[sl];B[mg];W[bm];B[am];W[lg]" \
              ";B[eq];W[es];B[mg];W[ha];B[gb];W[lg];B[ds];W[hs];B[mg];W[sj]" \
              ";B[si];W[lg];B[sr];W[sq];B[mg];W[hd];B[hb];W[lg];B[ro];W[so]" \
              ";B[mg];W[ss];B[qs];W[lg];B[sn];W[rp];B[mg];W[cl];B[bn];W[lg]" \
              ";B[ml];W[mk];B[mg];W[pj];B[sf];W[lg];B[nn];W[nl];B[mg];W[ib]" \
              ";B[ia];W[lg];B[nc];W[nb];B[mg];W[jd];B[kd];W[lg];B[ma];W[na]" \
              ";B[mg];W[qc];B[rc];W[lg];B[js];W[ks];B[mg];W[hc];B[id];W[lg]" \
              ";B[fk];W[hj];B[mg];W[hh];B[hg];W[lg];B[gk];W[hk];B[mg];W[ak]" \
              ";B[lg];W[al];B[bm];W[nf];B[od];W[ki];B[ms];W[kp];B[ip];W[jp]" \
              ";B[lr];W[oj];B[mr];W[ea];B[sr]))"

sgf_game = Sgf_game.from_string(sgf_content)

game_state = GameState.new_game(19)

for item in sgf_game.main_sequence_iter():
    color, move_tuple = item.get_move()
    if color is not None and move_tuple is not None:
        row, col = move_tuple
        point = Point(row + 1, col + 1)
        move = Move.play(point)
        game_state = game_state.apply_move(move)
        print_board(game_state.board)