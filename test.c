#include <stdio.h>

// Fungsi untuk Bubble Sort (ascending)
void bubble_sort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Fungsi menghitung poin maksimal
int calculate_max_points(int times[], int n, int x) {
    int time_used = 0, points = 0;
    for (int i = 0; i < n; i++) {
        if (time_used + times[i] <= x) {
            time_used += times[i];
            points++;
        } else {
            break;
        }
    }
    return points;
}

// Fungsi menghitung poin minimal
int calculate_min_points(int times[], int n, int x) {
    int time_used = 0, points = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (time_used + times[i] <= x) {
            time_used += times[i];
            points++;
        } else {
            break;
        }
    }
    return points;
}

int main() {
    int T;
    scanf("%d", &T);

    for (int t = 1; t <= T; t++) {
        int N, X;
        scanf("%d %d", &N, &X);
        int times[N];

        for (int i = 0; i < N; i++) {
            scanf("%d", &times[i]);
        }

        // Sorting waktu pengerjaan soal secara ascending
        bubble_sort(times, N);

        // Hitung poin minimal dan maksimal
        int min_points = calculate_min_points(times, N, X);
        int max_points = calculate_max_points(times, N, X);

        // Output hasil sesuai format
        printf("Case #%d: %d %d\n", t, min_points, max_points);
    }

    return 0;
}
