/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
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
public class LinkedBlockingDeque<E extends Object> extends AbstractQueue<E> implements BlockingDeque<E>, java.io.Serializable {

    @Positive
    static final class Node<E> {
    @Positive
    }

    @Positive
    public LinkedBlockingDeque() {
    @Positive
    }

    @Positive
    public LinkedBlockingDeque(int capacity) {
    @Positive
    }

    @Positive
    public LinkedBlockingDeque(Collection<? extends E> c) {
    @Positive
    }

    @Positive
    void unlink(@Shrinkable LinkedBlockingDeque<E> this, Node<E> x);

    @Positive
    public void addFirst(E e);

    @Positive
    public void addLast(E e);

    @Positive
    public boolean offerFirst(E e);

    @Positive
    public boolean offerLast(E e);

    @Positive
    public void putFirst(E e) throws InterruptedException;

    @Positive
    public void putLast(E e) throws InterruptedException;

    @Positive
    public boolean offerFirst(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public boolean offerLast(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E removeFirst(@GuardSatisfied @NonEmpty @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    public E removeLast(@GuardSatisfied @NonEmpty @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    @Nullable
    @Positive
    public E pollFirst(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    @Nullable
    @Positive
    public E pollLast(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    public E takeFirst(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this) throws InterruptedException;

    @Positive
    public E takeLast(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    public E pollFirst(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    public E pollLast(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E getFirst(@NonEmpty LinkedBlockingDeque<E> this);

    @Positive
    public E getLast(@NonEmpty LinkedBlockingDeque<E> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peekFirst();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peekLast();

    @Positive
    public boolean removeFirstOccurrence(@Shrinkable LinkedBlockingDeque<E> this, Object o);

    @Positive
    public boolean removeLastOccurrence(@Shrinkable LinkedBlockingDeque<E> this, Object o);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(E e);

    @Positive
    public boolean offer(E e);

    @Positive
    public void put(E e) throws InterruptedException;

    @Positive
    public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E remove(@GuardSatisfied @NonEmpty @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    public E take(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this) throws InterruptedException;

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E element(@NonEmpty LinkedBlockingDeque<E> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peek();

    @Positive
    public int remainingCapacity();

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this, Collection<? super E> c);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this, Collection<? super E> c, int maxElements);

    @Positive
    public void push(E e);

    @Positive
    public E pop(@GuardSatisfied @NonEmpty @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    public boolean remove(@Shrinkable LinkedBlockingDeque<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public boolean addAll(Collection<? extends E> c);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(LinkedBlockingDeque<@PolyNull @PolySigned E> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    public String toString();

    @Positive
    public void clear(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this);

    @Positive
    Node<E> succ(Node<E> p);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty LinkedBlockingDeque<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> descendingIterator(@PolyGrowShrink @PolyNonEmpty LinkedBlockingDeque<E> this);

    @Positive
    private abstract class AbstractItr implements Iterator<E> {

    @Positive
        abstract Node<E> firstNode();

    @Positive
        abstract Node<E> nextNode(Node<E> n);

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty AbstractItr this);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    private class Itr extends AbstractItr {

    @Positive
        Node<E> firstNode();

    @Positive
        Node<E> nextNode(Node<E> n);
    @Positive
    }

    @Positive
    private class DescendingItr extends AbstractItr {

    @Positive
        Node<E> firstNode();

    @Positive
        Node<E> nextNode(Node<E> n);
    @Positive
    }

    @Positive
    private final class LBDSpliterator implements Spliterator<E> {

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
    public boolean removeIf(@Shrinkable LinkedBlockingDeque<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@Shrinkable LinkedBlockingDeque<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable LinkedBlockingDeque<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    void checkInvariants();
    @Positive
}
