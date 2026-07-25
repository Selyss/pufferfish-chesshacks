// Print the engine's legal moves for a FEN, one per line, for cross-checking.
#include <cstdio>
#include <string>
#include <iostream>
#include "engine/types.h"
#include "engine/bitboard.h"
#include "engine/position.h"
#include "engine/movegen.h"
using namespace pf;
int main(){
    init_zobrist(); init_bitboards();
    std::string fen;
    while(std::getline(std::cin,fen)){
        if(fen.empty()) continue;
        Position p;
        if(!p.set_fen(fen)){ printf("BADFEN\n"); continue; }
        MoveList ml; generate_moves(p,ml); filter_legal_moves(p,ml);
        const char*f="abcdefgh";
        std::string out;
        for(int i=0;i<ml.count;++i){
            Move m=ml.moves[i];
            char b[8];
            const char*promo="";
            if(move_flags(m)&FLAG_PROMOTION){
                int pp=promo_piece(m);
                int t = (pp>=W_PAWN&&pp<=W_KING)?pp-W_PAWN:pp-B_PAWN;
                promo = (t==KNIGHT)?"n":(t==BISHOP)?"b":(t==ROOK)?"r":"q";
            }
            snprintf(b,sizeof b,"%c%d%c%d",f[from_sq(m)&7],(from_sq(m)>>3)+1,f[to_sq(m)&7],(to_sq(m)>>3)+1);
            out += std::string(b)+promo+" ";
        }
        printf("%s\n", out.c_str());
        fflush(stdout);
    }
    return 0;
}
