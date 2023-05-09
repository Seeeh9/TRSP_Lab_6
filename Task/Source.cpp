#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

double f(double x) {
    return sin(x);
}

double integrate_rectangle(double a, double b, int n) {
    double sum = 0.0;
    double dx = (b - a) / n;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        double x = a + (i + 0.5) * dx;
        sum += f(x) * dx;
    }
    return sum;
}

double integrate_trapezoid(double a, double b, int n) {
    double sum = (f(a) + f(b)) / 2.0;
    double dx = (b - a) / n;
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n; i++) {
        double x = a + i * dx;
        sum += f(x) * dx;
    }
    return sum;
}

double integrate_simpson(double a, double b, int n) {
    double sum = f(a) + f(b);
    double dx = (b - a) / n;
#pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n; i++) {
        double x = a + i * dx;
        if (i % 2 == 0)
            sum += 2.0 * f(x);
        else
            sum += 4.0 * f(x);
    }
    return sum * dx / 3.0;
}

int main() {
    double a, b;
    int n;

    cout << "Int_{a}^{b} f(x) dx \nFunction: Sin(x)\n";

    cout << "Enter a: ";
    cin >> a;

    cout << "Enter b: ";
    cin >> b;

    cout << "Enter amount of iterarions n: ";
    cin >> n;

    double start_time = omp_get_wtime();

    double result_rectangle = integrate_rectangle(a, b, n);
    double end_time_rectangle = omp_get_wtime();
    cout << "Rectangle method: " << result_rectangle << ", time: " << end_time_rectangle - start_time << endl;

    double start_time_trapezoid = omp_get_wtime();
    double result_trapezoid = integrate_trapezoid(a, b, n);
    double end_time_trapezoid = omp_get_wtime();
    cout << "Trapezoid method: " << result_trapezoid << ", time: " << end_time_trapezoid - start_time_trapezoid << endl;

    double start_time_simpson = omp_get_wtime();
    double result_simpson = integrate_simpson(a, b, n);
    double end_time_simpson = omp_get_wtime();
    cout << "Simpson method: " << result_simpson << ", time: " << end_time_simpson - start_time_simpson << endl;

    return 0;
}
