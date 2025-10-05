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
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.function.LongBinaryOperator;
import java.util.function.LongUnaryOperator;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class AtomicLongArray implements java.io.Serializable {

    public AtomicLongArray(int length) {
    }

    public AtomicLongArray(long[] array) {
    }

    public final int length();

    public final long get(int i);

    public final void set(int i, long newValue);

    public final void lazySet(int i, long newValue);

    public final long getAndSet(int i, long newValue);

    public final boolean compareAndSet(int i, long expectedValue, long newValue);

    @Deprecated()
    public final boolean weakCompareAndSet(int i, long expectedValue, long newValue);

    public final boolean weakCompareAndSetPlain(int i, long expectedValue, long newValue);

    public final long getAndIncrement(int i);

    public final long getAndDecrement(int i);

    public final long getAndAdd(int i, long delta);

    public final long incrementAndGet(int i);

    public final long decrementAndGet(int i);

    public long addAndGet(int i, long delta);

    public final long getAndUpdate(int i, LongUnaryOperator updateFunction);

    public final long updateAndGet(int i, LongUnaryOperator updateFunction);

    public final long getAndAccumulate(int i, long x, LongBinaryOperator accumulatorFunction);

    public final long accumulateAndGet(int i, long x, LongBinaryOperator accumulatorFunction);

    public String toString();

    public final long getPlain(int i);

    public final void setPlain(int i, long newValue);

    public final long getOpaque(int i);

    public final void setOpaque(int i, long newValue);

    public final long getAcquire(int i);

    public final void setRelease(int i, long newValue);

    public final long compareAndExchange(int i, long expectedValue, long newValue);

    public final long compareAndExchangeAcquire(int i, long expectedValue, long newValue);

    public final long compareAndExchangeRelease(int i, long expectedValue, long newValue);

    public final boolean weakCompareAndSetVolatile(int i, long expectedValue, long newValue);

    public final boolean weakCompareAndSetAcquire(int i, long expectedValue, long newValue);

    public final boolean weakCompareAndSetRelease(int i, long expectedValue, long newValue);
}
