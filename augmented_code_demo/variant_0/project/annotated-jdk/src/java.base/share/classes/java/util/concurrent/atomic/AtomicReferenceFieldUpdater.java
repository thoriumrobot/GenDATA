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
package java.util.concurrent.atomic;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import java.lang.invoke.VarHandle;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class AtomicReferenceFieldUpdater<T, V> {

    @Positive
    @CallerSensitive
    @Positive
    public static <U, W> AtomicReferenceFieldUpdater<U, W> newUpdater(Class<U> tclass, Class<W> vclass, String fieldName);

    @Positive
    protected AtomicReferenceFieldUpdater() {
    @Positive
    }

    @Positive
    public abstract boolean compareAndSet(T obj, V expect, V update);

    @Positive
    public abstract boolean weakCompareAndSet(T obj, V expect, V update);

    @Positive
    public abstract void set(T obj, V newValue);

    @Positive
    public abstract void lazySet(T obj, V newValue);

    @Positive
    public abstract V get(T obj);

    @Positive
    public V getAndSet(T obj, V newValue);

    @Positive
    public final V getAndUpdate(T obj, UnaryOperator<V> updateFunction);

    @Positive
    public final V updateAndGet(T obj, UnaryOperator<V> updateFunction);

    @Positive
    public final V getAndAccumulate(T obj, V x, BinaryOperator<V> accumulatorFunction);

    @Positive
    public final V accumulateAndGet(T obj, V x, BinaryOperator<V> accumulatorFunction);

    @Positive
    private static final class AtomicReferenceFieldUpdaterImpl<T, V> extends AtomicReferenceFieldUpdater<T, V> {

    @Positive
        static void throwCCE();

    @Positive
        public final boolean compareAndSet(T obj, V expect, V update);

    @Positive
        public final boolean weakCompareAndSet(T obj, V expect, V update);

    @Positive
        public final void set(T obj, V newValue);

    @Positive
        public final void lazySet(T obj, V newValue);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public final V get(T obj);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public final V getAndSet(T obj, V newValue);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
