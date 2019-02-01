#pragma once
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cmath>

template<int>
class Multigrid;

template<>
class Multigrid<0>
{
public:
    Multigrid(int, int){}
};

/*
mask == 0 - cell is live
mask != 0 - cell is dead, don't modify its values
value - current electric field value
charge - current charge
*/
template<int D>
class Multigrid
{
public:
    int w;
    int h;
    int* mask;
    float* values;
    float* bias;

    Multigrid<D-1> scaled;

    Multigrid(int w_, int h_)
    : w(w_)
    , h(h_)
    , mask(new int[w*h])
    , values(new float[w*h])
    , bias(new float[w*h])
    , scaled(w/2, h/2)
    {
        if ((w%2 || h%2) && D > 1)
            throw "Width or height not divisible by 2!";

        //initializing the values
        for (int x=0; x<w; x++)
            for (int y=0; y<h; y++)
            {
                int pos = y*w + x;
                mask[pos] = values[pos] = bias[pos] = 0;
            }

        //masking the border
        for (int x=0; x<w; x++)
        {
            mask[0*w + x] = 1;
            mask[(h-1)*w + x] = 1;
        }
        for (int y=0; y<h; y++)
        {
            mask[y*w + 0] = 1;
            mask[y*w + w-1] = 1;
        }
    }

    void recursion();

    void relax()
    {
        recursion();

        //relax here
        // int c = 0;
        for (int i=0; i<5; i++)
          step();
        // while(true)
        // {
        //     float err = step();
        //     if (err < 1e-10)
        //         break;
        //     c++;
        // }
        // printf("%d: %d\n", D, c);
    }

    float step()
    {
        float err = 0;
        for (int i=0; i<2; i++)
            #pragma omp parallel for collapse(2) if(w*h > 10000)
            for (int x=0; x<w; x++)
                for (int y=0; y<h; y++)
                    if (mask[y*w + x] == 0 && (x+y)%2 == i)
                    {
                        int pos = y*w + x;

                        float s = 0;
                        for (auto d : {-1, 1, -w, w})
                            s += values[pos + d];
                        s/=4;
                        s += bias[pos];
                        err += (values[pos]-s)*(values[pos]-s);
                        values[pos] = s;
                    }
        return err/(w*h);
    }

    void show(sf::RenderTarget& window, int id)
    {
        sf::Image img;
        img.create(w+20, h+20);
        for (int x=0; x<w; x++)
            for (int y=0; y<h; y++)
            {
                int idx = y*w + x;
                float v = values[idx];
                // int val = pow(v, 0.5)*255;
                int val = 255 + std::max(int(log(v)*10), -255);
                // int val = 255*v;
                val = std::min(val, 255);
                switch(mask[idx])
                {
                    case 0:
                        img.setPixel(x+10, y+10, sf::Color(val, val, val));
                        break;
                    case 1:
                        if (v > 0.5)
                            img.setPixel(x+10, y+10, sf::Color(77, 148, 255));
                        else
                            img.setPixel(x+10, y+10, sf::Color(255, 0, 0));
                        break;
                    case 2:
                        img.setPixel(x+10, y+10, sf::Color(51, 153, 51));
                        break;
                }
            }

        char fn[100];
        sprintf(fn, "lightning_%.3d.png", id);
        // printf("saved! %s\n", fn);
        // img.saveToFile(fn);

        sf::Texture tex;
        tex.loadFromImage(img);
        sf::Sprite sp(tex);
        sp.setScale(4, 4);
        window.draw(sp);
    }

    ~Multigrid()
    {
        delete[] mask;
        delete[] values;
        delete[] bias;
    }
};

template<>
void Multigrid<1>::recursion()
{}


/*
Main multigrid method.

1) copies the grid to the 2x smaller instance (averages value over 4 subpixels)
2) launches relaxation on the smaller instance
3) upscales the smaller instance back

*/
template<int D>
void Multigrid<D>::recursion()
{
    //copy scaled
    for (int x=0; x<w/2; x++)
        for (int y=0; y<h/2; y++)
        {
            int scaled_pos = y*scaled.w + x;
            int pos = 2*y*w + 2*x;

            scaled.mask[scaled_pos] = std::max(
                std::max(mask[pos], mask[pos+1]),
                std::max(mask[pos+w], mask[pos+w+1])
            );
            bool masked = scaled.mask[scaled_pos] > 0;

            int c = 0;
            scaled.values[scaled_pos] = scaled.bias[scaled_pos] = 0;
            for (auto d : {0, 1, w, w+1})
            {
                if (masked && mask[pos+d] == 0)
                    continue;
                scaled.values[scaled_pos] += values[pos+d];
                c++;
            }
            scaled.values[scaled_pos] /= c;
            scaled.bias[scaled_pos] = (bias[pos] + bias[pos+1] + bias[pos+w] + bias[pos+w+1])/4.f;
        }

    //relax scaled
    scaled.relax();

    //copy back unmasked values
    for (int x=0; x<w/2; x++)
        for (int y=0; y<h/2; y++)
        {
            int scaled_pos = y*scaled.w + x;
            int pos = 2*y*w + 2*x;
            for (auto d : {0, 1, w, w+1})
                if (mask[pos+d] == 0)
                    values[pos+d] = scaled.values[scaled_pos];
        }
}

