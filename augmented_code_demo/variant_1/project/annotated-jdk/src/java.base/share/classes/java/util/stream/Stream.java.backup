/*
* Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.LongConsumer;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;
import java.util.function.UnaryOperator;

@AnnotatedFor({ "lock", "mustcall", "nullness" })
@CFComment({ "MustCall: most Streams do not need to be closed.  There is no need for", "`@InheritableMustCall({})` because `AutoCloseable` already has that class annotation." })
public interface Stream<T> extends BaseStream<T, Stream<T>> {

    Stream<T> filter(Predicate<? super T> predicate);

    @PolyNonEmpty
    <R> Stream<R> map(@PolyNonEmpty Stream<T> this, Function<? super T, ? extends R> mapper);

    IntStream mapToInt(ToIntFunction<? super T> mapper);

    LongStream mapToLong(ToLongFunction<? super T> mapper);

    DoubleStream mapToDouble(ToDoubleFunction<? super T> mapper);

    <R> Stream<R> flatMap(Function<? super T, ? extends @Nullable Stream<? extends R>> mapper);

    IntStream flatMapToInt(Function<? super T, ? extends @Nullable IntStream> mapper);

    LongStream flatMapToLong(Function<? super T, ? extends @Nullable LongStream> mapper);

    DoubleStream flatMapToDouble(Function<? super T, ? extends @Nullable DoubleStream> mapper);

    default <R> Stream<R> mapMulti(BiConsumer<? super T, ? super Consumer<R>> mapper);

    default IntStream mapMultiToInt(BiConsumer<? super T, ? super IntConsumer> mapper);

    default LongStream mapMultiToLong(BiConsumer<? super T, ? super LongConsumer> mapper);

    default DoubleStream mapMultiToDouble(BiConsumer<? super T, ? super DoubleConsumer> mapper);

    @PolyNonEmpty
    Stream<T> distinct(@PolyNonEmpty Stream<T> this);

    @PolyNonEmpty
    Stream<T> sorted(@PolyNonEmpty Stream<T> this);

    @PolyNonEmpty
    Stream<T> sorted(@PolyNonEmpty Stream<T> this, Comparator<? super T> comparator);

    Stream<T> peek(Consumer<? super T> action);

    Stream<T> limit(long maxSize);

    Stream<T> skip(long n);

    default Stream<T> takeWhile(Predicate<? super T> predicate);

    default Stream<T> dropWhile(Predicate<? super T> predicate);

    void forEach(Consumer<? super T> action);

    void forEachOrdered(Consumer<? super T> action);

    @SideEffectFree
    @PolyNull
    Object @PolyNonEmpty [] toArray(@PolyNonEmpty Stream<@PolyNull T> this);

    @SideEffectFree
    <A> A[] toArray(IntFunction<A[]> generator);

    T reduce(T identity, BinaryOperator<T> accumulator);

    Optional<T> reduce(BinaryOperator<T> accumulator);

    <U> U reduce(U identity, BiFunction<U, ? super T, U> accumulator, BinaryOperator<U> combiner);

    @CFComment("@SideEffectFree: the supplied functions should not have side effects")
    @SideEffectFree
    <R> R collect(Supplier<R> supplier, BiConsumer<R, ? super T> accumulator, BiConsumer<R, R> combiner);

    @CFComment("@SideEffectFree: the collector should not have side effects")
    @SideEffectFree
    <R, A> R collect(Collector<? super T, A, R> collector);

    @SuppressWarnings("unchecked")
    default List<T> toList();

    Optional<T> min(Comparator<? super T> comparator);

    Optional<T> max(Comparator<? super T> comparator);

    long count();

    @EnsuresNonEmptyIf(result = true, expression = "this")
    boolean anyMatch(Stream<T> this, Predicate<? super T> predicate);

    @EnsuresNonEmptyIf(result = true, expression = "this")
    boolean allMatch(Stream<T> this, Predicate<? super T> predicate);

    @EnsuresNonEmptyIf(result = false, expression = "this")
    boolean noneMatch(Stream<T> this, Predicate<? super T> predicate);

    Optional<T> findFirst();

    Optional<T> findAny();

    public static <T> Builder<T> builder();

    public static <T> Stream<T> empty();

    @NonEmpty
    public static <T> Stream<T> of(T t);

    public static <T> Stream<T> ofNullable(@Nullable T t);

    @SafeVarargs
    @SuppressWarnings("varargs")
    @PolyNonEmpty
    public static <T> Stream<T> of(T@PolyNonEmpty ... values);

    public static <T> Stream<T> iterate(final T seed, final UnaryOperator<T> f);

    public static <T> Stream<T> iterate(T seed, Predicate<? super T> hasNext, UnaryOperator<T> next);

    public static <T> Stream<T> generate(Supplier<? extends T> s);

    public static <T> Stream<T> concat(Stream<? extends T> a, Stream<? extends T> b);

    public interface Builder<T> extends Consumer<T> {

        @Override
        @EnsuresNonEmpty("this")
        void accept(Stream.Builder<T> this, T t);

        @NonEmpty
        default Builder<T> add(Stream.@GuardSatisfied Builder<T> this, T t);

        @PolyNonEmpty
        Stream<T> build(Stream.@PolyNonEmpty Builder<T> this);
    }
}
