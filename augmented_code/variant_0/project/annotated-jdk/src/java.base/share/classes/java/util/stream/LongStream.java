/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2013, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.util.LongSummaryStatistics;
    @Positive
import java.util.Objects;
    @Positive
import java.util.OptionalDouble;
    @Positive
import java.util.OptionalLong;
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
import java.util.function.LongBinaryOperator;
    @Positive
import java.util.function.LongConsumer;
    @Positive
import java.util.function.LongFunction;
    @Positive
import java.util.function.LongPredicate;
    @Positive
import java.util.function.LongSupplier;
    @Positive
import java.util.function.LongToDoubleFunction;
    @Positive
import java.util.function.LongToIntFunction;
    @Positive
import java.util.function.LongUnaryOperator;
    @Positive
import java.util.function.ObjLongConsumer;
    @Positive
import java.util.function.Supplier;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface LongStream extends BaseStream<Long, LongStream> {

    @Positive
    LongStream filter(LongPredicate predicate);

    @Positive
    LongStream map(LongUnaryOperator mapper);

    @Positive
    <U> Stream<U> mapToObj(LongFunction<? extends U> mapper);

    @Positive
    IntStream mapToInt(LongToIntFunction mapper);

    @Positive
    DoubleStream mapToDouble(LongToDoubleFunction mapper);

    @Positive
    LongStream flatMap(LongFunction<? extends LongStream> mapper);

    @Positive
    default LongStream mapMulti(LongMapMultiConsumer mapper);

    @Positive
    LongStream distinct();

    @Positive
    LongStream sorted();

    @Positive
    LongStream peek(LongConsumer action);

    @Positive
    LongStream limit(long maxSize);

    @Positive
    LongStream skip(long n);

    @Positive
    default LongStream takeWhile(LongPredicate predicate);

    @Positive
    default LongStream dropWhile(LongPredicate predicate);

    @Positive
    void forEach(LongConsumer action);

    @Positive
    void forEachOrdered(LongConsumer action);

    @Positive
    @SideEffectFree
    @Positive
    long[] toArray();

    @Positive
    long reduce(long identity, LongBinaryOperator op);

    @Positive
    OptionalLong reduce(LongBinaryOperator op);

    @Positive
    <R> R collect(Supplier<R> supplier, ObjLongConsumer<R> accumulator, BiConsumer<R, R> combiner);

    @Positive
    long sum();

    @Positive
    OptionalLong min();

    @Positive
    OptionalLong max();

    @Positive
    long count();

    @Positive
    OptionalDouble average();

    @Positive
    LongSummaryStatistics summaryStatistics();

    @Positive
    boolean anyMatch(LongPredicate predicate);

    @Positive
    boolean allMatch(LongPredicate predicate);

    @Positive
    boolean noneMatch(LongPredicate predicate);

    @Positive
    OptionalLong findFirst();

    @Positive
    OptionalLong findAny();

    @Positive
    DoubleStream asDoubleStream();

    @Positive
    Stream<Long> boxed();

    @Positive
    @Override
    @Positive
    LongStream sequential();

    @Positive
    @Override
    @Positive
    LongStream parallel();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    PrimitiveIterator.OfLong iterator();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    Spliterator.OfLong spliterator();

    @Positive
    public static Builder builder();

    @Positive
    public static LongStream empty();

    @Positive
    public static LongStream of(long t);

    @Positive
    public static LongStream of(long... values);

    @Positive
    public static LongStream iterate(final long seed, final LongUnaryOperator f);

    @Positive
    public static LongStream iterate(long seed, LongPredicate hasNext, LongUnaryOperator next);

    @Positive
    public static LongStream generate(LongSupplier s);

    @Positive
    public static LongStream range(long startInclusive, final long endExclusive);

    @Positive
    public static LongStream rangeClosed(long startInclusive, final long endInclusive);

    @Positive
    public static LongStream concat(LongStream a, LongStream b);

    @Positive
    public interface Builder extends LongConsumer {

    @Positive
        @Override
    @Positive
        void accept(long t);

    @Positive
        default Builder add(LongStream.@GuardSatisfied Builder this, long t);

    @Positive
        LongStream build();
    @Positive
    }

    @Positive
    @FunctionalInterface
    @Positive
    interface LongMapMultiConsumer {

    @Positive
        void accept(long value, LongConsumer lc);
    @Positive
    }
    @Positive
}
