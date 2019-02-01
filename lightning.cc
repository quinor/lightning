#include "multigrid.hh"
#include <cmath>
#include <random>
#include <chrono>

const int W = 192;
const int H = 128;
// const int W = 1536;
// const int H = 1024;
const int SCALE = 4;
const float ETA = 2;
const float DIAG_SCALE = 0.7;
const float RES = 0.001;

using Dur = std::chrono::duration<double>;


Multigrid<4> sim(W, H);

int previous[W*H+7];

int main ()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);

    sf::RenderWindow window;
    sf::Event evt;

    window.create(
        sf::VideoMode(SCALE*W+80, SCALE*H+80, 32),
        "display",
        sf::Style::Titlebar | sf::Style::Close
        );



    //boundary conditions
    for (int x=0; x<W; x++)
    {
        sim.values[0*W + x] = 0;
        sim.values[(H-1)*W + x] = 1;

        int h1 = 0.25*H, h2 = 0.5*H, h3=0.75*H;

        if (x < 0.8*W)
        {
            sim.mask[h1*W + x] = sim.mask[h3*W + x] = 1;
            sim.values[h1*W + x] = sim.values[h3*W + x] = 0;
        }

        if (x > 0.2*W)
        {
            sim.mask[h2*W + x] = 1;
            sim.values[h2*W + x] = 0;
        }

    }
    for (int y=0; y<H; y++)
    {
        sim.values[y*W + 0] = 0;
        sim.values[y*W + W-1] = 0;
    }

    // seed
    sim.mask[1*W + W/2] = 2;
    sim.values[1*W + W/2] = 0;

    // float t_weightcomp, t_pointsplacing, t_postpro, t_relax;
    // t_weightcomp = t_pointsplacing = t_postpro = t_relax = 0;
    sim.relax();
    for (int frame=0;;)
    {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     //weight comp
        float wgt = 0;
        //#pragma omp parallel for collapse(2) reduction (+:wgt)
        for (int x=0; x<W; x++)
            for (int y=0; y<H; y++)
                if (sim.mask[y*W + x] == 0)
                {
                    int pos = y*W + x;
                    float cand = 0;
                    int prev = 0;

                    for (auto d : {W+1, W-1, -W-1, -W+1})
                        if (sim.mask[pos+d] == 2)
                        {
                            prev = pos+d;
                            cand = DIAG_SCALE;
                            break;
                        }

                    for (auto d : {-1, 1, -W, W})
                        if (sim.mask[pos+d] == 2)
                        {
                            prev = pos+d;
                            cand = 1;
                            break;
                        }
                    if (cand > 0)
                        sim.bias[pos] = 0;
                    previous[pos] = prev;
                    wgt += cand*pow(sim.values[pos], ETA);
                }

        wgt /= 1;

    //     auto t2 = std::chrono::high_resolution_clock::now();

        //points placing
        bool done = false;
        int last;
        #pragma omp parallel for collapse(2)
        for (int x=0; x<W; x++)
            for (int y=0; y<H; y++)
                if (sim.mask[y*W + x] == 0)
                {
                    int pos = y*W + x;
                    float cand = 0;

                    for (auto d : {-1, 1, -W, W})
                        if (sim.mask[pos+d] == 2)
                            cand = std::max(cand, 1.f);

                    for (auto d : {W+1, W-1, -W-1, -W+1})
                        if (sim.mask[pos+d] == 2)
                            cand = std::max(cand, DIAG_SCALE);

                    if (cand * pow(sim.values[pos], ETA)/wgt > dis(gen))
                    {
                        sim.mask[pos] = 3;
                        sim.values[pos] = 0;
                        for (auto d : {W+1, W-1, -W-1, -W+1})
                            if (sim.mask[pos+d] == 1 && sim.values[pos+d] > 0.5)
                            {
                                done = 1;
                                last = pos;
                            }
                    }
                }
        // auto t3 = std::chrono::high_resolution_clock::now();

        //postpro
        #pragma omp parallel for collapse(2)
        for (int x=0; x<W; x++)
            for (int y=0; y<H; y++)
                if (sim.mask[y*W + x] == 3)
                    sim.mask[y*W + x] = 2;


    //     auto t4 = std::chrono::high_resolution_clock::now();
        sim.relax();
    //     auto t5 = std::chrono::high_resolution_clock::now();
        window.display();
        while (window.pollEvent(evt))
            if(evt.type == sf::Event::Closed)
            {
                window.close();
                return 0;
            }

        sim.show(window, frame);
        if (done)
        {
            sim.show(window, frame);
            frame++;
            for (int x=0; x<W; x++)
                for (int y=0; y<H; y++)
                    if (sim.mask[y*W + x] != 1)
                    {
                        sim.mask[y*W + x] = sim.values[y*W + x] = 0;
                        if (sim.mask[y*W + x] != 0)
                            sim.bias[y*W + x] = RES;
                    }
            sim.mask[1*W + W/2] = 2;
            sim.values[1*W + W/2] = 0;
            int cur = last;
            while (cur)
            {
                sim.bias[cur] = RES*4;
                cur = previous[cur];
            }

        }
    //     const float SC = 0.8;
    //     t_weightcomp = t_weightcomp*SC + (1-SC)*Dur(t2-t1).count();
    //     t_pointsplacing = t_pointsplacing*SC + (1-SC)*Dur(t3-t2).count();
    //     t_postpro = t_postpro*SC + (1-SC)*Dur(t4-t3).count();
    //     t_relax = t_relax*SC + (1-SC)*Dur(t5-t4).count();
    //     printf("weightcomp: %f\npointsplacing: %f\npostpro: %f\nrelax: %f\n\n", t_weightcomp, t_pointsplacing, t_postpro, t_relax);
    }
}