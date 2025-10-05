/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1994, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.InheritableMustCall;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.util.ArraysSupport;

    @Positive
@AnnotatedFor({ "index", "lock", "mustcall", "nullness", "signedness" })
    @Positive
@InheritableMustCall({})
    @Positive
public class ByteArrayOutputStream extends OutputStream {

    @Positive
    protected byte[] buf;

    @Positive
    @IndexOrHigh({ "this.buf" })
    @Positive
    protected int count;

    @Positive
    public ByteArrayOutputStream() {
    @Positive
    }

    @Positive
    public ByteArrayOutputStream(@NonNegative int size) {
    @Positive
    }

    @Positive
    public synchronized void write(@PolySigned int b);

    @Positive
    public synchronized void write(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Positive
    public void writeBytes(byte[] b);

    @Positive
    public synchronized void writeTo(OutputStream out) throws IOException;

    @Positive
    public synchronized void reset();

    @Positive
    @PolySigned
    @Positive
    public synchronized byte[] toByteArray();

    @Positive
    @Pure
    @Positive
    @IndexOrHigh({ "this.buf" })
    @Positive
    public synchronized int size(@GuardSatisfied ByteArrayOutputStream this);

    @Positive
    @SideEffectFree
    @Positive
    public synchronized String toString(@GuardSatisfied ByteArrayOutputStream this);

    @Positive
    @SideEffectFree
    @Positive
    public synchronized String toString(@GuardSatisfied ByteArrayOutputStream this, String charsetName) throws UnsupportedEncodingException;

    @Positive
    public synchronized String toString(Charset charset);

    @Positive
    @SideEffectFree
    @Positive
    @Deprecated
    @Positive
    public synchronized String toString(@GuardSatisfied ByteArrayOutputStream this, int hibyte);

    @Positive
    public void close() throws IOException;
    @Positive
}
