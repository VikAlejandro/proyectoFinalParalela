#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;

// ------------------------------------------------------------
// CONFIGURACION 8K
// ------------------------------------------------------------
const int WIDTH = 7680;
const int HEIGHT = 4320;

const int MAX_ITER = 1000;

// ------------------------------------------------------------
// GUARDAR PPM
// ------------------------------------------------------------
void savePPM(const string& filename,
             const vector<unsigned char>& image)
{
    ofstream file(filename, ios::binary);

    file << "P6\n"
         << WIDTH << " "
         << HEIGHT
         << "\n255\n";

    file.write(reinterpret_cast<const char*>(image.data()),
               image.size());

    file.close();
}

// ------------------------------------------------------------
// GENERAR MANDELBROT
// ------------------------------------------------------------
void generateMandelbrot(vector<unsigned char>& image)
{
    double xmin = -2.5;
    double xmax = 1.0;
    double ymin = -1.0;
    double ymax = 1.0;

    cout << "\nGenerando fractal Mandelbrot 8K...\n";

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            double cx =
                xmin + (xmax - xmin) * x / WIDTH;

            double cy =
                ymin + (ymax - ymin) * y / HEIGHT;

            complex<double> c(cx, cy);
            complex<double> z(0, 0);

            int iter = 0;

            while (abs(z) <= 2.0 &&
                   iter < MAX_ITER)
            {
                z = z * z + c;
                iter++;
            }

            int idx = (y * WIDTH + x) * 3;

            unsigned char color =
                static_cast<unsigned char>(
                    255.0 * iter / MAX_ITER
                );

            image[idx]     = color;
            image[idx + 1] = color / 2;
            image[idx + 2] = 255 - color;
        }

        // ------------------------------------------------
        // MOSTRAR PROGRESO
        // ------------------------------------------------
        if (y % 100 == 0)
        {
            float percent =
                (100.0f * y) / HEIGHT;

            cout << "\rFractal: "
                 << percent
                 << "% completado"
                 << flush;
        }
    }

    cout << "\rFractal: 100% completado\n";
}

// ------------------------------------------------------------
// KERNEL GAUSSIANO
// ------------------------------------------------------------
vector<vector<double>> createGaussianKernel(
    int radius,
    double sigma)
{
    int size = 2 * radius + 1;

    vector<vector<double>> kernel(
        size,
        vector<double>(size)
    );

    double sum = 0.0;

    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            double value =
                exp(-(x * x + y * y) /
                (2 * sigma * sigma));

            kernel[y + radius][x + radius] =
                value;

            sum += value;
        }
    }

    // Normalizar
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            kernel[y][x] /= sum;
        }
    }

    return kernel;
}

// ------------------------------------------------------------
// BLUR GAUSSIANO PESADO
// ------------------------------------------------------------
void applyGaussianBlur(
    const vector<unsigned char>& input,
    vector<unsigned char>& output,
    int radius,
    double sigma)
{
    auto kernel =
        createGaussianKernel(radius, sigma);

    cout << "\nAplicando blur Gaussiano...\n";

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            double r = 0;
            double g = 0;
            double b = 0;

            for (int ky = -radius;
                 ky <= radius;
                 ky++)
            {
                for (int kx = -radius;
                     kx <= radius;
                     kx++)
                {
                    int ix =
                        min(max(x + kx, 0),
                            WIDTH - 1);

                    int iy =
                        min(max(y + ky, 0),
                            HEIGHT - 1);

                    int idx =
                        (iy * WIDTH + ix) * 3;

                    double weight =
                        kernel[ky + radius]
                              [kx + radius];

                    r += input[idx] * weight;
                    g += input[idx + 1] * weight;
                    b += input[idx + 2] * weight;
                }
            }

            int out =
                (y * WIDTH + x) * 3;

            output[out] =
                static_cast<unsigned char>(r);

            output[out + 1] =
                static_cast<unsigned char>(g);

            output[out + 2] =
                static_cast<unsigned char>(b);
        }

        // ------------------------------------------------
        // MOSTRAR PROGRESO
        // ------------------------------------------------
        if (y % 50 == 0)
        {
            float percent =
                (100.0f * y) / HEIGHT;

            cout << "\rBlur: "
                 << percent
                 << "% completado"
                 << flush;
        }
    }

    cout << "\rBlur: 100% completado\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    auto totalStart =
        chrono::high_resolution_clock::now();

    vector<unsigned char> image(
        WIDTH * HEIGHT * 3
    );

    vector<unsigned char> blurred(
        WIDTH * HEIGHT * 3
    );

    // --------------------------------------------------------
    // FRACTAL
    // --------------------------------------------------------
    auto fractalStart =
        chrono::high_resolution_clock::now();

    generateMandelbrot(image);

    auto fractalEnd =
        chrono::high_resolution_clock::now();

    savePPM("mandelbrot.ppm", image);

    cout << "Imagen fractal guardada.\n";

    // --------------------------------------------------------
    // BLUR
    // --------------------------------------------------------
    auto blurStart =
        chrono::high_resolution_clock::now();

    applyGaussianBlur(
        image,
        blurred,
        10,   // radio pesado
        8.0
    );

    auto blurEnd =
        chrono::high_resolution_clock::now();

    savePPM(
        "mandelbrot_blur.ppm",
        blurred
    );

    cout << "Imagen blur guardada.\n";

    // --------------------------------------------------------
    // TIEMPOS
    // --------------------------------------------------------
    chrono::duration<double> fractalTime =
        fractalEnd - fractalStart;

    chrono::duration<double> blurTime =
        blurEnd - blurStart;

    chrono::duration<double> totalTime =
        blurEnd - totalStart;

    cout << "\n=============================\n";
    cout << "Tiempo fractal: "
         << fractalTime.count()
         << " segundos\n";

    cout << "Tiempo blur: "
         << blurTime.count()
         << " segundos\n";

    cout << "Tiempo total: "
         << totalTime.count()
         << " segundos\n";

    cout << "=============================\n";

    return 0;
}