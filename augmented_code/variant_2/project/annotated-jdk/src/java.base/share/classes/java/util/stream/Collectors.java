/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2012, 2018, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.AbstractMap;
    @Positive
import java.util.AbstractSet;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.DoubleSummaryStatistics;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.IntSummaryStatistics;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.LongSummaryStatistics;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.function.ToDoubleFunction;
    @Positive
import java.util.function.ToIntFunction;
    @Positive
import java.util.function.ToLongFunction;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public final class Collectors {

    @Positive
    static class CollectorImpl<T, A, R> implements Collector<T, A, R> {

    @Positive
        @Override
    @Positive
        public BiConsumer<A, T> accumulator();

    @Positive
        @Override
    @Positive
        public Supplier<A> supplier();

    @Positive
        @Override
    @Positive
        public BinaryOperator<A> combiner();

    @Positive
        @Override
    @Positive
        public Function<A, R> finisher();

    @Positive
        @Override
    @Positive
        public Set<Characteristics> characteristics();
    @Positive
    }

    @Positive
    public static <T, C extends Collection<T>> Collector<T, ?, C> toCollection(Supplier<C> collectionFactory);

    @Positive
    @SideEffectFree
    @Positive
    public static <T> Collector<T, ?, List<T>> toList();

    @Positive
    public static <T> Collector<T, ?, List<T>> toUnmodifiableList();

    @Positive
    public static <T> Collector<T, ?, Set<T>> toSet();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> Collector<T, ?, Set<T>> toUnmodifiableSet();

    @Positive
    public static Collector<@Nullable CharSequence, ?, String> joining();

    @Positive
    public static Collector<@Nullable CharSequence, ?, String> joining(CharSequence delimiter);

    @Positive
    public static Collector<@Nullable CharSequence, ?, String> joining(CharSequence delimiter, CharSequence prefix, CharSequence suffix);

    @Positive
    public static <T, U, A, R> Collector<T, ?, R> mapping(Function<? super T, ? extends U> mapper, Collector<? super U, A, R> downstream);

    @Positive
    public static <T, U, A, R> Collector<T, ?, R> flatMapping(Function<? super T, ? extends Stream<? extends U>> mapper, Collector<? super U, A, R> downstream);

    @Positive
    public static <T, A, R> Collector<T, ?, R> filtering(Predicate<? super T> predicate, Collector<? super T, A, R> downstream);

    @Positive
    public static <T, A, R, RR> Collector<T, A, RR> collectingAndThen(Collector<T, A, R> downstream, Function<R, RR> finisher);

    @Positive
    public static <T> Collector<T, ?, Long> counting();

    @Positive
    public static <T> Collector<T, ?, Optional<T>> minBy(Comparator<? super T> comparator);

    @Positive
    public static <T> Collector<T, ?, Optional<T>> maxBy(Comparator<? super T> comparator);

    @Positive
    public static <T> Collector<T, ?, Integer> summingInt(ToIntFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, Long> summingLong(ToLongFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, Double> summingDouble(ToDoubleFunction<? super T> mapper);

    @Positive
    static double[] sumWithCompensation(double[] intermediateSum, double value);

    @Positive
    static double computeFinalSum(double[] summands);

    @Positive
    public static <T> Collector<T, ?, Double> averagingInt(ToIntFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, Double> averagingLong(ToLongFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, Double> averagingDouble(ToDoubleFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, T> reducing(T identity, BinaryOperator<T> op);

    @Positive
    public static <T> Collector<T, ?, Optional<T>> reducing(BinaryOperator<T> op);

    @Positive
    public static <T, U> Collector<T, ?, U> reducing(U identity, Function<? super T, ? extends U> mapper, BinaryOperator<U> op);

    @Positive
    public static <T, K> Collector<T, ?, Map<K, List<T>>> groupingBy(Function<? super T, ? extends K> classifier);

    @Positive
    public static <T, K, A, D> Collector<T, ?, Map<K, D>> groupingBy(Function<? super T, ? extends K> classifier, Collector<? super T, A, D> downstream);

    @Positive
    public static <T, K, D, A, M extends Map<K, D>> Collector<T, ?, M> groupingBy(Function<? super T, ? extends K> classifier, Supplier<M> mapFactory, Collector<? super T, A, D> downstream);

    @Positive
    public static <T, K extends Object> Collector<T, ?, ConcurrentMap<K, List<T>>> groupingByConcurrent(Function<? super T, ? extends K> classifier);

    @Positive
    public static <T, K extends Object, A, D extends Object> Collector<T, ?, ConcurrentMap<K, D>> groupingByConcurrent(Function<? super T, ? extends K> classifier, Collector<? super T, A, D> downstream);

    @Positive
    public static <T, K extends Object, A, D extends Object, M extends ConcurrentMap<K, D>> Collector<T, ?, M> groupingByConcurrent(Function<? super T, ? extends K> classifier, Supplier<M> mapFactory, Collector<? super T, A, D> downstream);

    @Positive
    public static <T> Collector<T, ?, Map<Boolean, List<T>>> partitioningBy(Predicate<? super T> predicate);

    @Positive
    public static <T, D, A> Collector<T, ?, Map<Boolean, D>> partitioningBy(Predicate<? super T> predicate, Collector<? super T, A, D> downstream);

    @Positive
    public static <T, K, U extends Object> Collector<T, ?, Map<K, U>> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    public static <T, K extends Object, U extends Object> Collector<T, ?, Map<K, U>> toUnmodifiableMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper);

    @Positive
    public static <T, K, U extends Object> Collector<T, ?, Map<K, U>> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    public static <T, K extends Object, U extends Object> Collector<T, ?, Map<K, U>> toUnmodifiableMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction);

    @Positive
    public static <T, K, U extends Object, M extends Map<K, U>> Collector<T, ?, M> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction, Supplier<M> mapFactory);

    @Positive
    public static <T, K extends Object, U extends Object> Collector<T, ?, ConcurrentMap<K, U>> toConcurrentMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper);

    @Positive
    public static <T, K extends Object, U extends Object> Collector<T, ?, ConcurrentMap<K, U>> toConcurrentMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction);

    @Positive
    public static <T, K extends Object, U extends Object, M extends ConcurrentMap<K, U>> Collector<T, ?, M> toConcurrentMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction, Supplier<M> mapFactory);

    @Positive
    public static <T> Collector<T, ?, IntSummaryStatistics> summarizingInt(ToIntFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, LongSummaryStatistics> summarizingLong(ToLongFunction<? super T> mapper);

    @Positive
    public static <T> Collector<T, ?, DoubleSummaryStatistics> summarizingDouble(ToDoubleFunction<? super T> mapper);

    @Positive
    public static <T, R1, R2, R> Collector<T, ?, R> teeing(Collector<? super T, ?, R1> downstream1, Collector<? super T, ?, R2> downstream2, BiFunction<? super R1, ? super R2, R> merger);

    @Positive
    private static final class Partition<T> extends AbstractMap<Boolean, T> implements Map<Boolean, T> {

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<Boolean, T>> entrySet();
    @Positive
    }
    @Positive
}
