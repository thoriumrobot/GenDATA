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
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;
import jdk.internal.misc.Unsafe;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import java.lang.invoke.VarHandle;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class AtomicIntegerFieldUpdater<T> {

    @CallerSensitive
    public static <U> AtomicIntegerFieldUpdater<U> newUpdater(Class<U> tclass, String fieldName);

    protected AtomicIntegerFieldUpdater() {
    }

    public abstract boolean compareAndSet(T obj, int expect, int update);

    public abstract boolean weakCompareAndSet(T obj, int expect, int update);

    public abstract void set(T obj, int newValue);

    public abstract void lazySet(T obj, int newValue);

    public abstract int get(T obj);

    public int getAndSet(T obj, int newValue);

    public int getAndIncrement(T obj);

    public int getAndDecrement(T obj);

    public int getAndAdd(T obj, int delta);

    public int incrementAndGet(T obj);

    public int decrementAndGet(T obj);

    public int addAndGet(T obj, int delta);

    public final int getAndUpdate(T obj, IntUnaryOperator updateFunction);

    public final int updateAndGet(T obj, IntUnaryOperator updateFunction);

    public final int getAndAccumulate(T obj, int x, IntBinaryOperator accumulatorFunction);

    public final int accumulateAndGet(T obj, int x, IntBinaryOperator accumulatorFunction);

    private static final class AtomicIntegerFieldUpdaterImpl<T> extends AtomicIntegerFieldUpdater<T> {

        public final boolean compareAndSet(T obj, int expect, int update);

        public final boolean weakCompareAndSet(T obj, int expect, int update);

        public final void set(T obj, int newValue);

        public final void lazySet(T obj, int newValue);

        public final int get(T obj);

        public final int getAndSet(T obj, int newValue);

        public final int getAndAdd(T obj, int delta);

        public final int getAndIncrement(T obj);

        public final int getAndDecrement(T obj);

        public final int incrementAndGet(T obj);

        public final int decrementAndGet(T obj);

        public final int addAndGet(T obj, int delta);
    }
}
