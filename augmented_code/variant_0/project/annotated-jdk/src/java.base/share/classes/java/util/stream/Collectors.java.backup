/*
 * Copyright (c) 2012, 2018, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util.stream;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.DoubleSummaryStatistics;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IntSummaryStatistics;
import java.util.Iterator;
import java.util.List;
import java.util.LongSummaryStatistics;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;
import jdk.internal.access.SharedSecrets;

@AnnotatedFor({ "lock", "nullness" })
public final class Collectors {

    static class CollectorImpl<T, A, R> implements Collector<T, A, R> {

        @Override
        public BiConsumer<A, T> accumulator();

        @Override
        public Supplier<A> supplier();

        @Override
        public BinaryOperator<A> combiner();

        @Override
        public Function<A, R> finisher();

        @Override
        public Set<Characteristics> characteristics();
    }

    public static <T, C extends Collection<T>> Collector<T, ?, C> toCollection(Supplier<C> collectionFactory);

    @SideEffectFree
    public static <T> Collector<T, ?, List<T>> toList();

    public static <T> Collector<T, ?, List<T>> toUnmodifiableList();

    public static <T> Collector<T, ?, Set<T>> toSet();

    @SuppressWarnings("unchecked")
    public static <T> Collector<T, ?, Set<T>> toUnmodifiableSet();

    public static Collector<@Nullable CharSequence, ?, String> joining();

    public static Collector<@Nullable CharSequence, ?, String> joining(CharSequence delimiter);

    public static Collector<@Nullable CharSequence, ?, String> joining(CharSequence delimiter, CharSequence prefix, CharSequence suffix);

    public static <T, U, A, R> Collector<T, ?, R> mapping(Function<? super T, ? extends U> mapper, Collector<? super U, A, R> downstream);

    public static <T, U, A, R> Collector<T, ?, R> flatMapping(Function<? super T, ? extends Stream<? extends U>> mapper, Collector<? super U, A, R> downstream);

    public static <T, A, R> Collector<T, ?, R> filtering(Predicate<? super T> predicate, Collector<? super T, A, R> downstream);

    public static <T, A, R, RR> Collector<T, A, RR> collectingAndThen(Collector<T, A, R> downstream, Function<R, RR> finisher);

    public static <T> Collector<T, ?, Long> counting();

    public static <T> Collector<T, ?, Optional<T>> minBy(Comparator<? super T> comparator);

    public static <T> Collector<T, ?, Optional<T>> maxBy(Comparator<? super T> comparator);

    public static <T> Collector<T, ?, Integer> summingInt(ToIntFunction<? super T> mapper);

    public static <T> Collector<T, ?, Long> summingLong(ToLongFunction<? super T> mapper);

    public static <T> Collector<T, ?, Double> summingDouble(ToDoubleFunction<? super T> mapper);

    static double[] sumWithCompensation(double[] intermediateSum, double value);

    static double computeFinalSum(double[] summands);

    public static <T> Collector<T, ?, Double> averagingInt(ToIntFunction<? super T> mapper);

    public static <T> Collector<T, ?, Double> averagingLong(ToLongFunction<? super T> mapper);

    public static <T> Collector<T, ?, Double> averagingDouble(ToDoubleFunction<? super T> mapper);

    public static <T> Collector<T, ?, T> reducing(T identity, BinaryOperator<T> op);

    public static <T> Collector<T, ?, Optional<T>> reducing(BinaryOperator<T> op);

    public static <T, U> Collector<T, ?, U> reducing(U identity, Function<? super T, ? extends U> mapper, BinaryOperator<U> op);

    public static <T, K> Collector<T, ?, Map<K, List<T>>> groupingBy(Function<? super T, ? extends K> classifier);

    public static <T, K, A, D> Collector<T, ?, Map<K, D>> groupingBy(Function<? super T, ? extends K> classifier, Collector<? super T, A, D> downstream);

    public static <T, K, D, A, M extends Map<K, D>> Collector<T, ?, M> groupingBy(Function<? super T, ? extends K> classifier, Supplier<M> mapFactory, Collector<? super T, A, D> downstream);

    public static <T, K extends Object> Collector<T, ?, ConcurrentMap<K, List<T>>> groupingByConcurrent(Function<? super T, ? extends K> classifier);

    public static <T, K extends Object, A, D extends Object> Collector<T, ?, ConcurrentMap<K, D>> groupingByConcurrent(Function<? super T, ? extends K> classifier, Collector<? super T, A, D> downstream);

    public static <T, K extends Object, A, D extends Object, M extends ConcurrentMap<K, D>> Collector<T, ?, M> groupingByConcurrent(Function<? super T, ? extends K> classifier, Supplier<M> mapFactory, Collector<? super T, A, D> downstream);

    public static <T> Collector<T, ?, Map<Boolean, List<T>>> partitioningBy(Predicate<? super T> predicate);

    public static <T, D, A> Collector<T, ?, Map<Boolean, D>> partitioningBy(Predicate<? super T> predicate, Collector<? super T, A, D> downstream);

    public static <T, K, U extends Object> Collector<T, ?, Map<K, U>> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper);

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public static <T, K extends Object, U extends Object> Collector<T, ?, Map<K, U>> toUnmodifiableMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper);

    public static <T, K, U extends Object> Collector<T, ?, Map<K, U>> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction);

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public static <T, K extends Object, U extends Object> Collector<T, ?, Map<K, U>> toUnmodifiableMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction);

    public static <T, K, U extends Object, M extends Map<K, U>> Collector<T, ?, M> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction, Supplier<M> mapFactory);

    public static <T, K extends Object, U extends Object> Collector<T, ?, ConcurrentMap<K, U>> toConcurrentMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper);

    public static <T, K extends Object, U extends Object> Collector<T, ?, ConcurrentMap<K, U>> toConcurrentMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction);

    public static <T, K extends Object, U extends Object, M extends ConcurrentMap<K, U>> Collector<T, ?, M> toConcurrentMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, BinaryOperator<U> mergeFunction, Supplier<M> mapFactory);

    public static <T> Collector<T, ?, IntSummaryStatistics> summarizingInt(ToIntFunction<? super T> mapper);

    public static <T> Collector<T, ?, LongSummaryStatistics> summarizingLong(ToLongFunction<? super T> mapper);

    public static <T> Collector<T, ?, DoubleSummaryStatistics> summarizingDouble(ToDoubleFunction<? super T> mapper);

    public static <T, R1, R2, R> Collector<T, ?, R> teeing(Collector<? super T, ?, R1> downstream1, Collector<? super T, ?, R2> downstream2, BiFunction<? super R1, ? super R2, R> merger);

    private static final class Partition<T> extends AbstractMap<Boolean, T> implements Map<Boolean, T> {

        @Override
        @SideEffectFree
        public Set<Map.Entry<Boolean, T>> entrySet();
    }
}
