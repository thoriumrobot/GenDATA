/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.AbstractQueue;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.concurrent.locks.Condition;
    @Positive
import java.util.concurrent.locks.ReentrantLock;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class LinkedBlockingQueue<E extends Object> extends AbstractQueue<E> implements BlockingQueue<E>, java.io.Serializable {

    @Positive
    static class Node<E> {
    @Positive
    }

    @Positive
    void fullyLock();

    @Positive
    void fullyUnlock();

    @Positive
    public LinkedBlockingQueue() {
    @Positive
    }

    @Positive
    public LinkedBlockingQueue(int capacity) {
    @Positive
    }

    @Positive
    public LinkedBlockingQueue(Collection<? extends E> c) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    public int remainingCapacity();

    @Positive
    public void put(E e) throws InterruptedException;

    @Positive
    public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public boolean offer(E e);

    @Positive
    public E take(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peek();

    @Positive
    void unlink(Node<E> p, Node<E> pred);

    @Positive
    public boolean remove(@Shrinkable LinkedBlockingQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(LinkedBlockingQueue<@PolyNull @PolySigned E> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    public String toString();

    @Positive
    public void clear(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this, Collection<? super E> c);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this, Collection<? super E> c, int maxElements);

    @Positive
    Node<E> succ(Node<E> p);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty LinkedBlockingQueue<E> this);

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
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    private final class LBQSpliterator implements Spliterator<E> {

    @Positive
        public long estimateSize();

    @Positive
        public Spliterator<E> trySplit();

    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public Spliterator<E> spliterator();

    @Positive
    public void forEach(Consumer<? super E> action);

    @Positive
    void forEachFrom(Consumer<? super E> action, Node<E> p);

    @Positive
    public boolean removeIf(@Shrinkable LinkedBlockingQueue<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@Shrinkable LinkedBlockingQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable LinkedBlockingQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    Node<E> findPred(Node<E> p, Node<E> ancestor);
    @Positive
}
