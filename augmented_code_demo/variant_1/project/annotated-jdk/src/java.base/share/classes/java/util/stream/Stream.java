/*
    @Positive
* Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.util.*;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.DoubleConsumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.function.IntFunction;
    @Positive
import java.util.function.LongConsumer;
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
import java.util.function.UnaryOperator;

    @Positive
@AnnotatedFor({ "lock", "mustcall", "nullness" })
    @Positive
@CFComment({ "MustCall: most Streams do not need to be closed.  There is no need for", "`@InheritableMustCall({})` because `AutoCloseable` already has that class annotation." })
    @Positive
public interface Stream<T> extends BaseStream<T, Stream<T>> {

    @Positive
    Stream<T> filter(Predicate<? super T> predicate);

    @Positive
    @PolyNonEmpty
    @Positive
    <R> Stream<R> map(@PolyNonEmpty Stream<T> this, Function<? super T, ? extends R> mapper);

    @Positive
    IntStream mapToInt(ToIntFunction<? super T> mapper);

    @Positive
    LongStream mapToLong(ToLongFunction<? super T> mapper);

    @Positive
    DoubleStream mapToDouble(ToDoubleFunction<? super T> mapper);

    @Positive
    <R> Stream<R> flatMap(Function<? super T, ? extends @Nullable Stream<? extends R>> mapper);

    @Positive
    IntStream flatMapToInt(Function<? super T, ? extends @Nullable IntStream> mapper);

    @Positive
    LongStream flatMapToLong(Function<? super T, ? extends @Nullable LongStream> mapper);

    @Positive
    DoubleStream flatMapToDouble(Function<? super T, ? extends @Nullable DoubleStream> mapper);

    @Positive
    default <R> Stream<R> mapMulti(BiConsumer<? super T, ? super Consumer<R>> mapper);

    @Positive
    default IntStream mapMultiToInt(BiConsumer<? super T, ? super IntConsumer> mapper);

    @Positive
    default LongStream mapMultiToLong(BiConsumer<? super T, ? super LongConsumer> mapper);

    @Positive
    default DoubleStream mapMultiToDouble(BiConsumer<? super T, ? super DoubleConsumer> mapper);

    @Positive
    @PolyNonEmpty
    @Positive
    Stream<T> distinct(@PolyNonEmpty Stream<T> this);

    @Positive
    @PolyNonEmpty
    @Positive
    Stream<T> sorted(@PolyNonEmpty Stream<T> this);

    @Positive
    @PolyNonEmpty
    @Positive
    Stream<T> sorted(@PolyNonEmpty Stream<T> this, Comparator<? super T> comparator);

    @Positive
    Stream<T> peek(Consumer<? super T> action);

    @Positive
    Stream<T> limit(long maxSize);

    @Positive
    Stream<T> skip(long n);

    @Positive
    default Stream<T> takeWhile(Predicate<? super T> predicate);

    @Positive
    default Stream<T> dropWhile(Predicate<? super T> predicate);

    @Positive
    void forEach(Consumer<? super T> action);

    @Positive
    void forEachOrdered(Consumer<? super T> action);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    Object @PolyNonEmpty [] toArray(@PolyNonEmpty Stream<@PolyNull T> this);

    @Positive
    @SideEffectFree
    @Positive
    <A> A[] toArray(IntFunction<A[]> generator);

    @Positive
    T reduce(T identity, BinaryOperator<T> accumulator);

    @Positive
    Optional<T> reduce(BinaryOperator<T> accumulator);

    @Positive
    <U> U reduce(U identity, BiFunction<U, ? super T, U> accumulator, BinaryOperator<U> combiner);

    @Positive
    @CFComment("@SideEffectFree: the supplied functions should not have side effects")
    @Positive
    @SideEffectFree
    @Positive
    <R> R collect(Supplier<R> supplier, BiConsumer<R, ? super T> accumulator, BiConsumer<R, R> combiner);

    @Positive
    @CFComment("@SideEffectFree: the collector should not have side effects")
    @Positive
    @SideEffectFree
    @Positive
    <R, A> R collect(Collector<? super T, A, R> collector);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    default List<T> toList();

    @Positive
    Optional<T> min(Comparator<? super T> comparator);

    @Positive
    Optional<T> max(Comparator<? super T> comparator);

    @Positive
    long count();

    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean anyMatch(Stream<T> this, Predicate<? super T> predicate);

    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    boolean allMatch(Stream<T> this, Predicate<? super T> predicate);

    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    boolean noneMatch(Stream<T> this, Predicate<? super T> predicate);

    @Positive
    Optional<T> findFirst();

    @Positive
    Optional<T> findAny();

    @Positive
    public static <T> Builder<T> builder();

    @Positive
    public static <T> Stream<T> empty();

    @Positive
    @NonEmpty
    @Positive
    public static <T> Stream<T> of(T t);

    @Positive
    public static <T> Stream<T> ofNullable(@Nullable T t);

    @Positive
    @SafeVarargs
    @Positive
    @SuppressWarnings("varargs")
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> Stream<T> of(T@PolyNonEmpty ... values);

    @Positive
    public static <T> Stream<T> iterate(final T seed, final UnaryOperator<T> f);

    @Positive
    public static <T> Stream<T> iterate(T seed, Predicate<? super T> hasNext, UnaryOperator<T> next);

    @Positive
    public static <T> Stream<T> generate(Supplier<? extends T> s);

    @Positive
    public static <T> Stream<T> concat(Stream<? extends T> a, Stream<? extends T> b);

    @Positive
    public interface Builder<T> extends Consumer<T> {

    @Positive
        @Override
    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        void accept(Stream.Builder<T> this, T t);

    @Positive
        @NonEmpty
    @Positive
        default Builder<T> add(Stream.@GuardSatisfied Builder<T> this, T t);

    @Positive
        @PolyNonEmpty
    @Positive
        Stream<T> build(Stream.@PolyNonEmpty Builder<T> this);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
