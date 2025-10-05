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
import java.util.function.IntBinaryOperator;
    @Positive
import java.util.function.IntUnaryOperator;
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
public abstract class AtomicIntegerFieldUpdater<T> {

    @Positive
    @CallerSensitive
    @Positive
    public static <U> AtomicIntegerFieldUpdater<U> newUpdater(Class<U> tclass, String fieldName);

    @Positive
    protected AtomicIntegerFieldUpdater() {
    @Positive
    }

    @Positive
    public abstract boolean compareAndSet(T obj, int expect, int update);

    @Positive
    public abstract boolean weakCompareAndSet(T obj, int expect, int update);

    @Positive
    public abstract void set(T obj, int newValue);

    @Positive
    public abstract void lazySet(T obj, int newValue);

    @Positive
    public abstract int get(T obj);

    @Positive
    public int getAndSet(T obj, int newValue);

    @Positive
    public int getAndIncrement(T obj);

    @Positive
    public int getAndDecrement(T obj);

    @Positive
    public int getAndAdd(T obj, int delta);

    @Positive
    public int incrementAndGet(T obj);

    @Positive
    public int decrementAndGet(T obj);

    @Positive
    public int addAndGet(T obj, int delta);

    @Positive
    public final int getAndUpdate(T obj, IntUnaryOperator updateFunction);

    @Positive
    public final int updateAndGet(T obj, IntUnaryOperator updateFunction);

    @Positive
    public final int getAndAccumulate(T obj, int x, IntBinaryOperator accumulatorFunction);

    @Positive
    public final int accumulateAndGet(T obj, int x, IntBinaryOperator accumulatorFunction);

    @Positive
    private static final class AtomicIntegerFieldUpdaterImpl<T> extends AtomicIntegerFieldUpdater<T> {

    @Positive
        public final boolean compareAndSet(T obj, int expect, int update);

    @Positive
        public final boolean weakCompareAndSet(T obj, int expect, int update);

    @Positive
        public final void set(T obj, int newValue);

    @Positive
        public final void lazySet(T obj, int newValue);

    @Positive
        public final int get(T obj);

    @Positive
        public final int getAndSet(T obj, int newValue);

    @Positive
        public final int getAndAdd(T obj, int delta);

    @Positive
        public final int getAndIncrement(T obj);

    @Positive
        public final int getAndDecrement(T obj);

    @Positive
        public final int incrementAndGet(T obj);

    @Positive
        public final int decrementAndGet(T obj);

    @Positive
        public final int addAndGet(T obj, int delta);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
