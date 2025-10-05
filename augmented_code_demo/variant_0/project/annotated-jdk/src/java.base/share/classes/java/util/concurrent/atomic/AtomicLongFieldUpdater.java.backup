/*
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
package java.util.concurrent.atomic;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.function.LongBinaryOperator;
import java.util.function.LongUnaryOperator;
import jdk.internal.misc.Unsafe;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import java.lang.invoke.VarHandle;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class AtomicLongFieldUpdater<T> {

    @CallerSensitive
    public static <U> AtomicLongFieldUpdater<U> newUpdater(Class<U> tclass, String fieldName);

    protected AtomicLongFieldUpdater() {
    }

    public abstract boolean compareAndSet(T obj, long expect, long update);

    public abstract boolean weakCompareAndSet(T obj, long expect, long update);

    public abstract void set(T obj, long newValue);

    public abstract void lazySet(T obj, long newValue);

    public abstract long get(T obj);

    public long getAndSet(T obj, long newValue);

    public long getAndIncrement(T obj);

    public long getAndDecrement(T obj);

    public long getAndAdd(T obj, long delta);

    public long incrementAndGet(T obj);

    public long decrementAndGet(T obj);

    public long addAndGet(T obj, long delta);

    public final long getAndUpdate(T obj, LongUnaryOperator updateFunction);

    public final long updateAndGet(T obj, LongUnaryOperator updateFunction);

    public final long getAndAccumulate(T obj, long x, LongBinaryOperator accumulatorFunction);

    public final long accumulateAndGet(T obj, long x, LongBinaryOperator accumulatorFunction);

    private static final class CASUpdater<T> extends AtomicLongFieldUpdater<T> {

        public final boolean compareAndSet(T obj, long expect, long update);

        public final boolean weakCompareAndSet(T obj, long expect, long update);

        public final void set(T obj, long newValue);

        public final void lazySet(T obj, long newValue);

        public final long get(T obj);

        public final long getAndSet(T obj, long newValue);

        public final long getAndAdd(T obj, long delta);

        public final long getAndIncrement(T obj);

        public final long getAndDecrement(T obj);

        public final long incrementAndGet(T obj);

        public final long decrementAndGet(T obj);

        public final long addAndGet(T obj, long delta);
    }

    private static final class LockedUpdater<T> extends AtomicLongFieldUpdater<T> {

        public final boolean compareAndSet(T obj, long expect, long update);

        public final boolean weakCompareAndSet(T obj, long expect, long update);

        public final void set(T obj, long newValue);

        public final void lazySet(T obj, long newValue);

        public final long get(T obj);
    }

    static boolean isAncestor(ClassLoader first, ClassLoader second);

    static boolean isSamePackage(Class<?> class1, Class<?> class2);
}
