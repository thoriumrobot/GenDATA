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
import java.util.function.LongBinaryOperator;
    @Positive
import java.util.function.LongUnaryOperator;
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
public abstract class AtomicLongFieldUpdater<T> {

    @Positive
    @CallerSensitive
    @Positive
    public static <U> AtomicLongFieldUpdater<U> newUpdater(Class<U> tclass, String fieldName);

    @Positive
    protected AtomicLongFieldUpdater() {
    @Positive
    }

    @Positive
    public abstract boolean compareAndSet(T obj, long expect, long update);

    @Positive
    public abstract boolean weakCompareAndSet(T obj, long expect, long update);

    @Positive
    public abstract void set(T obj, long newValue);

    @Positive
    public abstract void lazySet(T obj, long newValue);

    @Positive
    public abstract long get(T obj);

    @Positive
    public long getAndSet(T obj, long newValue);

    @Positive
    public long getAndIncrement(T obj);

    @Positive
    public long getAndDecrement(T obj);

    @Positive
    public long getAndAdd(T obj, long delta);

    @Positive
    public long incrementAndGet(T obj);

    @Positive
    public long decrementAndGet(T obj);

    @Positive
    public long addAndGet(T obj, long delta);

    @Positive
    public final long getAndUpdate(T obj, LongUnaryOperator updateFunction);

    @Positive
    public final long updateAndGet(T obj, LongUnaryOperator updateFunction);

    @Positive
    public final long getAndAccumulate(T obj, long x, LongBinaryOperator accumulatorFunction);

    @Positive
    public final long accumulateAndGet(T obj, long x, LongBinaryOperator accumulatorFunction);

    @Positive
    private static final class CASUpdater<T> extends AtomicLongFieldUpdater<T> {

    @Positive
        public final boolean compareAndSet(T obj, long expect, long update);

    @Positive
        public final boolean weakCompareAndSet(T obj, long expect, long update);

    @Positive
        public final void set(T obj, long newValue);

    @Positive
        public final void lazySet(T obj, long newValue);

    @Positive
        public final long get(T obj);

    @Positive
        public final long getAndSet(T obj, long newValue);

    @Positive
        public final long getAndAdd(T obj, long delta);

    @Positive
        public final long getAndIncrement(T obj);

    @Positive
        public final long getAndDecrement(T obj);

    @Positive
        public final long incrementAndGet(T obj);

    @Positive
        public final long decrementAndGet(T obj);

    @Positive
        public final long addAndGet(T obj, long delta);
    @Positive
    }

    @Positive
    private static final class LockedUpdater<T> extends AtomicLongFieldUpdater<T> {

    @Positive
        public final boolean compareAndSet(T obj, long expect, long update);

    @Positive
        public final boolean weakCompareAndSet(T obj, long expect, long update);

    @Positive
        public final void set(T obj, long newValue);

    @Positive
        public final void lazySet(T obj, long newValue);

    @Positive
        public final long get(T obj);
    @Positive
    }

    @Positive
    static boolean isAncestor(ClassLoader first, ClassLoader second);

    @Positive
    static boolean isSamePackage(Class<?> class1, Class<?> class2);
    @Positive
}

// CFWR semantic augmentation - variant 1
