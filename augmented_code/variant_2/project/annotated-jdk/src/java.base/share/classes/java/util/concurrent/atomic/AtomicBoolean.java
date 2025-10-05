/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AtomicBoolean implements java.io.Serializable {

    @Positive
    public AtomicBoolean(boolean initialValue) {
    @Positive
    }

    @Positive
    public AtomicBoolean() {
    @Positive
    }

    @Positive
    public final boolean get();

    @Positive
    public final boolean compareAndSet(boolean expectedValue, boolean newValue);

    @Positive
    @Deprecated()
    @Positive
    public boolean weakCompareAndSet(boolean expectedValue, boolean newValue);

    @Positive
    public boolean weakCompareAndSetPlain(boolean expectedValue, boolean newValue);

    @Positive
    public final void set(boolean newValue);

    @Positive
    public final void lazySet(boolean newValue);

    @Positive
    public final boolean getAndSet(boolean newValue);

    @Positive
    public String toString();

    @Positive
    public final boolean getPlain();

    @Positive
    public final void setPlain(boolean newValue);

    @Positive
    public final boolean getOpaque();

    @Positive
    public final void setOpaque(boolean newValue);

    @Positive
    public final boolean getAcquire();

    @Positive
    public final void setRelease(boolean newValue);

    @Positive
    public final boolean compareAndExchange(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean compareAndExchangeAcquire(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean compareAndExchangeRelease(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean weakCompareAndSetVolatile(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean weakCompareAndSetAcquire(boolean expectedValue, boolean newValue);

    @Positive
    public final boolean weakCompareAndSetRelease(boolean expectedValue, boolean newValue);
    @Positive
}
