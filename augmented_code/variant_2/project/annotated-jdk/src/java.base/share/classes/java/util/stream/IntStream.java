/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util.stream;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.IntSummaryStatistics;
    @Positive
import java.util.Objects;
    @Positive
import java.util.OptionalDouble;
    @Positive
import java.util.OptionalInt;
    @Positive
import java.util.PrimitiveIterator;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.IntBinaryOperator;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.function.IntFunction;
    @Positive
import java.util.function.IntPredicate;
    @Positive
import java.util.function.IntSupplier;
    @Positive
import java.util.function.IntToDoubleFunction;
    @Positive
import java.util.function.IntToLongFunction;
    @Positive
import java.util.function.IntUnaryOperator;
    @Positive
import java.util.function.ObjIntConsumer;
    @Positive
import java.util.function.Supplier;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface IntStream extends BaseStream<Integer, IntStream> {

    @Positive
    IntStream filter(IntPredicate predicate);

    @Positive
    IntStream map(IntUnaryOperator mapper);

    @Positive
    <U> Stream<U> mapToObj(IntFunction<? extends U> mapper);

    @Positive
    LongStream mapToLong(IntToLongFunction mapper);

    @Positive
    DoubleStream mapToDouble(IntToDoubleFunction mapper);

    @Positive
    IntStream flatMap(IntFunction<? extends IntStream> mapper);

    @Positive
    default IntStream mapMulti(IntMapMultiConsumer mapper);

    @Positive
    IntStream distinct();

    @Positive
    IntStream sorted();

    @Positive
    IntStream peek(IntConsumer action);

    @Positive
    IntStream limit(long maxSize);

    @Positive
    IntStream skip(long n);

    @Positive
    default IntStream takeWhile(IntPredicate predicate);

    @Positive
    default IntStream dropWhile(IntPredicate predicate);

    @Positive
    void forEach(IntConsumer action);

    @Positive
    void forEachOrdered(IntConsumer action);

    @Positive
    @SideEffectFree
    @Positive
    int[] toArray();

    @Positive
    int reduce(int identity, IntBinaryOperator op);

    @Positive
    OptionalInt reduce(IntBinaryOperator op);

    @Positive
    <R> R collect(Supplier<R> supplier, ObjIntConsumer<R> accumulator, BiConsumer<R, R> combiner);

    @Positive
    int sum();

    @Positive
    OptionalInt min();

    @Positive
    OptionalInt max();

    @Positive
    long count();

    @Positive
    OptionalDouble average();

    @Positive
    IntSummaryStatistics summaryStatistics();

    @Positive
    boolean anyMatch(IntPredicate predicate);

    @Positive
    boolean allMatch(IntPredicate predicate);

    @Positive
    boolean noneMatch(IntPredicate predicate);

    @Positive
    OptionalInt findFirst();

    @Positive
    OptionalInt findAny();

    @Positive
    LongStream asLongStream();

    @Positive
    DoubleStream asDoubleStream();

    @Positive
    Stream<Integer> boxed();

    @Positive
    @Override
    @Positive
    IntStream sequential();

    @Positive
    @Override
    @Positive
    IntStream parallel();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    PrimitiveIterator.OfInt iterator();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    Spliterator.OfInt spliterator();

    @Positive
    public static Builder builder();

    @Positive
    public static IntStream empty();

    @Positive
    public static IntStream of(int t);

    @Positive
    public static IntStream of(int... values);

    @Positive
    public static IntStream iterate(final int seed, final IntUnaryOperator f);

    @Positive
    public static IntStream iterate(int seed, IntPredicate hasNext, IntUnaryOperator next);

    @Positive
    public static IntStream generate(IntSupplier s);

    @Positive
    public static IntStream range(int startInclusive, int endExclusive);

    @Positive
    public static IntStream rangeClosed(int startInclusive, int endInclusive);

    @Positive
    public static IntStream concat(IntStream a, IntStream b);

    @Positive
    public interface Builder extends IntConsumer {

    @Positive
        @Override
    @Positive
        void accept(int t);

    @Positive
        default Builder add(IntStream.@GuardSatisfied Builder this, int t);

    @Positive
        IntStream build();
    @Positive
    }

    @Positive
    @FunctionalInterface
    @Positive
    interface IntMapMultiConsumer {

    @Positive
        void accept(int value, IntConsumer ic);
    @Positive
    }
    @Positive
}
