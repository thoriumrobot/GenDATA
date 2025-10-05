/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2002, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import sun.nio.cs.HistoricallyNamedCharset;
    @Positive
import sun.nio.cs.*;

    @Positive
public class ISO2022_CN_CNS extends ISO2022 implements HistoricallyNamedCharset {

    @Positive
    public ISO2022_CN_CNS() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean contains(Charset cs);

    @Positive
    public String historicalName();

    @Positive
    public CharsetDecoder newDecoder();

    @Positive
    public CharsetEncoder newEncoder();

    @Positive
    private static class Encoder extends ISO2022.Encoder {

    @Positive
        public Encoder(Charset cs) {
    @Positive
        }

    @Positive
        public boolean canEncode(char c);

    @Positive
        public boolean isLegalReplacement(byte[] repl);
    @Positive
    }
    @Positive
}
