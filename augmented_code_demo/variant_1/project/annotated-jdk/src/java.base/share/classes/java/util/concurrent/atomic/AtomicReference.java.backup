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
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class AtomicReference<V> implements java.io.Serializable {

    public AtomicReference(V initialValue) {
    }

    public AtomicReference() {
    }

    public final V get();

    public final void set(V newValue);

    public final void lazySet(V newValue);

    public final boolean compareAndSet(V expectedValue, V newValue);

    @Deprecated()
    public final boolean weakCompareAndSet(V expectedValue, V newValue);

    public final boolean weakCompareAndSetPlain(V expectedValue, V newValue);

    @SuppressWarnings("unchecked")
    public final V getAndSet(V newValue);

    public final V getAndUpdate(UnaryOperator<V> updateFunction);

    public final V updateAndGet(UnaryOperator<V> updateFunction);

    public final V getAndAccumulate(V x, BinaryOperator<V> accumulatorFunction);

    public final V accumulateAndGet(V x, BinaryOperator<V> accumulatorFunction);

    public String toString();

    public final V getPlain();

    public final void setPlain(V newValue);

    public final V getOpaque();

    public final void setOpaque(V newValue);

    public final V getAcquire();

    public final void setRelease(V newValue);

    public final V compareAndExchange(V expectedValue, V newValue);

    public final V compareAndExchangeAcquire(V expectedValue, V newValue);

    public final V compareAndExchangeRelease(V expectedValue, V newValue);

    public final boolean weakCompareAndSetVolatile(V expectedValue, V newValue);

    public final boolean weakCompareAndSetAcquire(V expectedValue, V newValue);

    public final boolean weakCompareAndSetRelease(V expectedValue, V newValue);
}
