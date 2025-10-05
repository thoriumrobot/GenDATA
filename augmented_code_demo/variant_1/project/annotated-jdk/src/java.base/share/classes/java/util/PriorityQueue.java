/*
    @Positive
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.index.qual.Shrinkable;
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
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.util.ArraysSupport;

    @Positive
@CFComment({ "lock/nullness: This class doesn't permits null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
@SuppressWarnings("unchecked")
    @Positive
public class PriorityQueue<E extends @NonNull Object> extends AbstractQueue<E> implements java.io.Serializable {

    @Positive
    public PriorityQueue() {
    @Positive
    }

    @Positive
    public PriorityQueue(@Positive int initialCapacity) {
    @Positive
    }

    @Positive
    public PriorityQueue(Comparator<? super E> comparator) {
    @Positive
    }

    @Positive
    public PriorityQueue(@Positive int initialCapacity, Comparator<? super E> comparator) {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public PriorityQueue(@PolyNonEmpty Collection<? extends E> c) {
    @Positive
    }

    @Positive
    public PriorityQueue(PriorityQueue<? extends E> c) {
    @Positive
    }

    @Positive
    public PriorityQueue(SortedSet<? extends E> c) {
    @Positive
    }

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(@GuardSatisfied PriorityQueue<E> this, E e);

    @Positive
    public boolean offer(E e);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peek(@GuardSatisfied PriorityQueue<E> this);

    @Positive
    public boolean remove(@GuardSatisfied @Shrinkable PriorityQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    void removeEq(@GuardSatisfied @Shrinkable PriorityQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied PriorityQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(PriorityQueue<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty PriorityQueue<E> this);

    @Positive
    private final class Itr implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty Itr this);

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied PriorityQueue<E> this);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable PriorityQueue<E> this);

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable PriorityQueue<E> this);

    @Positive
    E removeAt(@GuardSatisfied @NonEmpty @Shrinkable PriorityQueue<E> this, int i);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Comparator<? super E> comparator(@GuardSatisfied PriorityQueue<E> this);

    @Positive
    public final Spliterator<E> spliterator();

    @Positive
    final class PriorityQueueSpliterator implements Spliterator<E> {

    @Positive
        public PriorityQueueSpliterator trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public boolean removeIf(@GuardSatisfied @Shrinkable PriorityQueue<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@GuardSatisfied @Shrinkable PriorityQueue<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable PriorityQueue<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public void forEach(Consumer<? super E> action);
    @Positive
}

// CFWR semantic augmentation - variant 1
