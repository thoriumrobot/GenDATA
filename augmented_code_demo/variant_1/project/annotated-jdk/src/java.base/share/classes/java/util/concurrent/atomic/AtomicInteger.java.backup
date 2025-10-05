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
import java.lang.invoke.VarHandle;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;
import jdk.internal.misc.Unsafe;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class AtomicInteger extends Number implements java.io.Serializable {

    public AtomicInteger(int initialValue) {
    }

    public AtomicInteger() {
    }

    public final int get();

    public final void set(int newValue);

    public final void lazySet(int newValue);

    public final int getAndSet(int newValue);

    public final boolean compareAndSet(int expectedValue, int newValue);

    @Deprecated()
    public final boolean weakCompareAndSet(int expectedValue, int newValue);

    public final boolean weakCompareAndSetPlain(int expectedValue, int newValue);

    public final int getAndIncrement();

    public final int getAndDecrement();

    public final int getAndAdd(int delta);

    public final int incrementAndGet();

    public final int decrementAndGet();

    public final int addAndGet(int delta);

    public final int getAndUpdate(IntUnaryOperator updateFunction);

    public final int updateAndGet(IntUnaryOperator updateFunction);

    public final int getAndAccumulate(int x, IntBinaryOperator accumulatorFunction);

    public final int accumulateAndGet(int x, IntBinaryOperator accumulatorFunction);

    public String toString();

    public int intValue();

    public long longValue();

    public float floatValue();

    public double doubleValue();

    public final int getPlain();

    public final void setPlain(int newValue);

    public final int getOpaque();

    public final void setOpaque(int newValue);

    public final int getAcquire();

    public final void setRelease(int newValue);

    public final int compareAndExchange(int expectedValue, int newValue);

    public final int compareAndExchangeAcquire(int expectedValue, int newValue);

    public final int compareAndExchangeRelease(int expectedValue, int newValue);

    public final boolean weakCompareAndSetVolatile(int expectedValue, int newValue);

    public final boolean weakCompareAndSetAcquire(int expectedValue, int newValue);

    public final boolean weakCompareAndSetRelease(int expectedValue, int newValue);
}
