/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
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
import java.util.DoubleSummaryStatistics;
    @Positive
import java.util.Objects;
    @Positive
import java.util.OptionalDouble;
    @Positive
import java.util.PrimitiveIterator;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.DoubleBinaryOperator;
    @Positive
import java.util.function.DoubleConsumer;
    @Positive
import java.util.function.DoubleFunction;
    @Positive
import java.util.function.DoublePredicate;
    @Positive
import java.util.function.DoubleSupplier;
    @Positive
import java.util.function.DoubleToIntFunction;
    @Positive
import java.util.function.DoubleToLongFunction;
    @Positive
import java.util.function.DoubleUnaryOperator;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.ObjDoubleConsumer;
    @Positive
import java.util.function.Supplier;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface DoubleStream extends BaseStream<Double, DoubleStream> {

    @Positive
    DoubleStream filter(DoublePredicate predicate);

    @Positive
    DoubleStream map(DoubleUnaryOperator mapper);

    @Positive
    <U> Stream<U> mapToObj(DoubleFunction<? extends U> mapper);

    @Positive
    IntStream mapToInt(DoubleToIntFunction mapper);

    @Positive
    LongStream mapToLong(DoubleToLongFunction mapper);

    @Positive
    DoubleStream flatMap(DoubleFunction<? extends DoubleStream> mapper);

    @Positive
    default DoubleStream mapMulti(DoubleMapMultiConsumer mapper);

    @Positive
    DoubleStream distinct();

    @Positive
    DoubleStream sorted();

    @Positive
    DoubleStream peek(DoubleConsumer action);

    @Positive
    DoubleStream limit(long maxSize);

    @Positive
    DoubleStream skip(long n);

    @Positive
    default DoubleStream takeWhile(DoublePredicate predicate);

    @Positive
    default DoubleStream dropWhile(DoublePredicate predicate);

    @Positive
    void forEach(DoubleConsumer action);

    @Positive
    void forEachOrdered(DoubleConsumer action);

    @Positive
    @SideEffectFree
    @Positive
    double[] toArray();

    @Positive
    double reduce(double identity, DoubleBinaryOperator op);

    @Positive
    OptionalDouble reduce(DoubleBinaryOperator op);

    @Positive
    <R> R collect(Supplier<R> supplier, ObjDoubleConsumer<R> accumulator, BiConsumer<R, R> combiner);

    @Positive
    double sum();

    @Positive
    OptionalDouble min();

    @Positive
    OptionalDouble max();

    @Positive
    long count();

    @Positive
    OptionalDouble average();

    @Positive
    DoubleSummaryStatistics summaryStatistics();

    @Positive
    boolean anyMatch(DoublePredicate predicate);

    @Positive
    boolean allMatch(DoublePredicate predicate);

    @Positive
    boolean noneMatch(DoublePredicate predicate);

    @Positive
    OptionalDouble findFirst();

    @Positive
    OptionalDouble findAny();

    @Positive
    Stream<Double> boxed();

    @Positive
    @Override
    @Positive
    DoubleStream sequential();

    @Positive
    @Override
    @Positive
    DoubleStream parallel();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    PrimitiveIterator.OfDouble iterator();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    Spliterator.OfDouble spliterator();

    @Positive
    public static Builder builder();

    @Positive
    public static DoubleStream empty();

    @Positive
    public static DoubleStream of(double t);

    @Positive
    public static DoubleStream of(double... values);

    @Positive
    public static DoubleStream iterate(final double seed, final DoubleUnaryOperator f);

    @Positive
    public static DoubleStream iterate(double seed, DoublePredicate hasNext, DoubleUnaryOperator next);

    @Positive
    public static DoubleStream generate(DoubleSupplier s);

    @Positive
    public static DoubleStream concat(DoubleStream a, DoubleStream b);

    @Positive
    public interface Builder extends DoubleConsumer {

    @Positive
        @Override
    @Positive
        void accept(double t);

    @Positive
        default Builder add(DoubleStream.@GuardSatisfied Builder this, double t);

    @Positive
        DoubleStream build();
    @Positive
    }

    @Positive
    @FunctionalInterface
    @Positive
    interface DoubleMapMultiConsumer {

    @Positive
        void accept(double value, DoubleConsumer dc);
    @Positive
    }
    @Positive
}
