#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// ============================================================
// CONFIGURACION
// ============================================================
const int WIDTH = 7680;
const int HEIGHT = 4320;

const int MAX_ITER = 1000;

// ============================================================
// GUARDAR PPM
// ============================================================
void savePPM(const string& filename,
             const vector<unsigned char>& image)
{
    ofstream file(filename, ios::binary);

    file << "P6\n"
         << WIDTH << " "
         << HEIGHT
         << "\n255\n";

    file.write(
        reinterpret_cast<const char*>(image.data()),
        image.size()
    );

    file.close();
}

// ============================================================
// CREAR KERNEL GAUSSIANO
// ============================================================
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
                (2.0 * sigma * sigma));

            kernel[y + radius][x + radius] =
                value;

            sum += value;
        }
    }

    // --------------------------------------------------------
    // NORMALIZAR KERNEL
    // --------------------------------------------------------
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            kernel[y][x] /= sum;
        }
    }

    return kernel;
}

// ============================================================
// MANDELBROT PARALELO
// Scheduler ganador:
// dynamic + chunk size 50
// ============================================================
void generateMandelbrot(
    vector<unsigned char>& image)
{
    double xmin = -2.5;
    double xmax = 1.0;
    double ymin = -1.0;
    double ymax = 1.0;

    omp_set_schedule(
        omp_sched_dynamic,
        50
    );

    cout << "\n=================================\n";
    cout << "GENERANDO MANDELBROT 8K\n";
    cout << "Scheduler: dynamic\n";
    cout << "Chunk size: 50\n";
    cout << "=================================\n";

    auto start =
        chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // OPENMP
    // --------------------------------------------------------
    #pragma omp parallel for schedule(runtime)
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

        // ----------------------------------------------------
        // PROGRESO
        // ----------------------------------------------------
        if (omp_get_thread_num() == 0 &&
            y % 100 == 0)
        {
            float percent =
                (100.0f * y) / HEIGHT;

            cout << "\rFractal: "
                 << percent
                 << "%"
                 << flush;
        }
    }

    auto end =
        chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed =
        end - start;

    cout << "\rFractal: 100%\n";

    cout << "Tiempo fractal: "
         << elapsed.count()
         << " segundos\n";
}

// ============================================================
// FILTRO GAUSSIANO PARALELO + SIMD
// ============================================================
void applyGaussianBlur(
    const vector<unsigned char>& input,
    vector<unsigned char>& output,
    int radius,
    double sigma)
{
    cout << "\n=================================\n";
    cout << "FILTRO GAUSSIANO PARALELO SIMD\n";
    cout << "Radius: " << radius << endl;
    cout << "Sigma: " << sigma << endl;
    cout << "=================================\n";

    auto kernel =
        createGaussianKernel(radius, sigma);

    omp_set_schedule(
        omp_sched_dynamic,
        50
    );

    auto start =
        chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // OPENMP
    // --------------------------------------------------------
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;

            // ------------------------------------------------
            // CONVOLUCION
            // ------------------------------------------------
            for (int ky = -radius;
                 ky <= radius;
                 ky++)
            {
                // --------------------------------------------
                // SIMD / SPMD
                // --------------------------------------------
                #pragma omp simd reduction(+:r,g,b)
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

        // ----------------------------------------------------
        // PROGRESO
        // ----------------------------------------------------
        if (omp_get_thread_num() == 0 &&
            y % 50 == 0)
        {
            float percent =
                (100.0f * y) / HEIGHT;

            cout << "\rBlur: "
                 << percent
                 << "%"
                 << flush;
        }
    }

    auto end =
        chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed =
        end - start;

    cout << "\rBlur: 100%\n";

    cout << "Tiempo blur SIMD: "
         << elapsed.count()
         << " segundos\n";
}

// ============================================================
// HISTOGRAMA CON ATOMIC
// ============================================================
void histogramAtomic(
    const vector<unsigned char>& image)
{
    vector<int> histogram(256, 0);

    omp_set_schedule(
        omp_sched_dynamic,
        50
    );

    cout << "\n=================================\n";
    cout << "HISTOGRAMA ATOMIC\n";
    cout << "=================================\n";

    auto start =
        chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(runtime)
    for (long long i = 0;
         i < WIDTH * HEIGHT * 3;
         i += 3)
    {
        int intensity = image[i];

        // ----------------------------------------------------
        // EXCLUSION MUTUA
        // ----------------------------------------------------
        #pragma omp atomic
        histogram[intensity]++;
    }

    auto end =
        chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed =
        end - start;

    cout << "Tiempo histogram atomic: "
         << elapsed.count()
         << " segundos\n";

    cout << "\nPrimeros colores:\n";

    for (int i = 0; i < 10; i++)
    {
        cout << "Color "
             << i
             << ": "
             << histogram[i]
             << endl;
    }
}

// ============================================================
// HISTOGRAMA LOCAL PRIVADO
// ============================================================
void histogramLocal(
    const vector<unsigned char>& image)
{
    vector<int> histogram(256, 0);

    omp_set_schedule(
        omp_sched_dynamic,
        50
    );

    cout << "\n=================================\n";
    cout << "HISTOGRAMA LOCAL PRIVADO\n";
    cout << "=================================\n";

    auto start =
        chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // ----------------------------------------------------
        // HISTOGRAMA PRIVADO
        // ----------------------------------------------------
        vector<int> localHistogram(256, 0);

        #pragma omp for schedule(runtime)
        for (long long i = 0;
             i < WIDTH * HEIGHT * 3;
             i += 3)
        {
            int intensity = image[i];

            localHistogram[intensity]++;
        }

        // ----------------------------------------------------
        // COMBINAR RESULTADOS
        // ----------------------------------------------------
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++)
            {
                histogram[i] +=
                    localHistogram[i];
            }
        }
    }

    auto end =
        chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed =
        end - start;

    cout << "Tiempo histogram local: "
         << elapsed.count()
         << " segundos\n";

    cout << "\nPrimeros colores:\n";

    for (int i = 0; i < 10; i++)
    {
        cout << "Color "
             << i
             << ": "
             << histogram[i]
             << endl;
    }
}

// ============================================================
// MAIN
// ============================================================
int main()
{
    cout << "=================================\n";
    cout << "MANDELBROT 8K HPC\n";
    cout << "OpenMP Threads: "
         << omp_get_max_threads()
         << endl;
    cout << "=================================\n";

    auto totalStart =
        chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // MEMORIA
    // --------------------------------------------------------
    vector<unsigned char> image(
        WIDTH * HEIGHT * 3
    );

    vector<unsigned char> blurred(
        WIDTH * HEIGHT * 3
    );

    // --------------------------------------------------------
    // MANDELBROT
    // --------------------------------------------------------
    generateMandelbrot(image);

    savePPM(
        "mandelbrot_8k.ppm",
        image
    );

    cout << "Imagen fractal guardada.\n";

    // --------------------------------------------------------
    // BLUR GAUSSIANO SIMD
    // --------------------------------------------------------
    applyGaussianBlur(
        image,
        blurred,
        35,      // radio grande
        20.0     // sigma grande
    );

    savePPM(
        "mandelbrot_blur_8k.ppm",
        blurred
    );

    cout << "Imagen blur guardada.\n";

    // --------------------------------------------------------
    // HISTOGRAMAS
    // --------------------------------------------------------
    histogramAtomic(blurred);

    histogramLocal(blurred);

    auto totalEnd =
        chrono::high_resolution_clock::now();

    chrono::duration<double> total =
        totalEnd - totalStart;

    cout << "\n=================================\n";
    cout << "TIEMPO TOTAL: "
         << total.count()
         << " segundos\n";
    cout << "=================================\n";

    return 0;
}