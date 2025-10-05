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
