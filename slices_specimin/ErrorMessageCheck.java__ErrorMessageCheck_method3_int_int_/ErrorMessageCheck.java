import org.checkerframework.checker.index.qual.NonNegative;

public class ErrorMessageCheck {

    void method3(@NonNegative int size, @NonNegative int value) {
        this.size = size;
        this.vDown = new int[this.size];
        vDown[1 + value] = 10;
    }
}
