/*
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
package java.util.concurrent;

    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
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
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.util.AbstractQueue;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Queue;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class ConcurrentLinkedQueue<E extends @NonNull Object> extends AbstractQueue<E> implements Queue<E>, java.io.Serializable {

    @Positive
    static final class Node<E> {

    @Positive
        void appendRelaxed(Node<E> next);

    @Positive
        boolean casItem(E cmp, E val);
    @Positive
    }

    @Positive
    public ConcurrentLinkedQueue() {
    @Positive
    }

    @Positive
    public ConcurrentLinkedQueue(Collection<? extends E> c) {
    @Positive
    }

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(E e);

    @Positive
    final void updateHead(Node<E> h, Node<E> p);

    @Positive
    final Node<E> succ(Node<E> p);

    @Positive
    public boolean offer(E e);

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable ConcurrentLinkedQueue<E> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peek();

    @Positive
    @Nullable
    @Positive
    Node<E> first();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @UnknownSignedness Object o);

    @Positive
    public boolean remove(@Shrinkable ConcurrentLinkedQueue<E> this, @GuardSatisfied @UnknownSignedness Object o);

    @Positive
    public boolean addAll(Collection<? extends E> c);

    @Positive
    public String toString();

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(ConcurrentLinkedQueue<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
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
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty ConcurrentLinkedQueue<E> this);

    @Positive
    private class Itr implements Iterator<E> {

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
    final class CLQSpliterator implements Spliterator<E> {

    @Positive
        public Spliterator<E> trySplit();

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
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Spliterator<E> spliterator();

    @Positive
    public boolean removeIf(@Shrinkable ConcurrentLinkedQueue<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@Shrinkable ConcurrentLinkedQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable ConcurrentLinkedQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable ConcurrentLinkedQueue<E> this);

    @Positive
    void forEachFrom(Consumer<? super E> action, Node<E> p);

    @Positive
    public void forEach(Consumer<? super E> action);
    @Positive
}

// CFWR semantic augmentation - variant 1
