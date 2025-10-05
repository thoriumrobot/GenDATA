/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.EOFException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.FilterOutputStream;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.ByteArrayOutputStream;

    @Positive
@AnnotatedFor({ "mustcall", "signedness" })
    @Positive
public class DigestOutputStream extends FilterOutputStream {

    @Positive
    protected MessageDigest digest;

    @Positive
    @MustCallAlias
    @Positive
    public DigestOutputStream(@MustCallAlias OutputStream stream, MessageDigest digest) {
    @Positive
    }

    @Positive
    public MessageDigest getMessageDigest();

    @Positive
    public void setMessageDigest(MessageDigest digest);

    @Positive
    public void write(int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Positive
    public void on(boolean on);

    @Positive
    public String toString();
    @Positive
}
