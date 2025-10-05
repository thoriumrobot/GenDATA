/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.cs.ext;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import sun.nio.cs.DoubleByte;
    @Positive
import sun.nio.cs.*;
    @Positive
import static sun.nio.cs.CharsetMapping.*;

    @Positive
public class MS932_0213 extends Charset {

    @Positive
    public MS932_0213() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean contains(Charset cs);

    @Positive
    public CharsetDecoder newDecoder();

    @Positive
    public CharsetEncoder newEncoder();

    @Positive
    protected static class Decoder extends SJIS_0213.Decoder {

    @Positive
        protected Decoder(Charset cs) {
    @Positive
        }

    @Positive
        protected char decodeDouble(int b1, int b2);
    @Positive
    }

    @Positive
    protected static class Encoder extends SJIS_0213.Encoder {

    @Positive
        protected Encoder(Charset cs) {
    @Positive
        }

    @Positive
        protected int encodeChar(char ch);
    @Positive
    }
    @Positive
}
