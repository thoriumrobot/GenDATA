/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Date;

    @Positive
public class DerInputStream {

    @Positive
    public DerInputStream(byte[] data, int start, int length, boolean allowBER) {
    @Positive
    }

    @Positive
    public DerInputStream(byte[] data) throws IOException {
    @Positive
    }

    @Positive
    public DerInputStream(byte[] data, int offset, int len) throws IOException {
    @Positive
    }

    @Positive
    public byte[] toByteArray();

    @Positive
    public DerValue getDerValue() throws IOException;

    @Positive
    public int getInteger() throws IOException;

    @Positive
    public BigInteger getBigInteger() throws IOException;

    @Positive
    public BigInteger getPositiveBigInteger() throws IOException;

    @Positive
    public int getEnumerated() throws IOException;

    @Positive
    public byte[] getBitString() throws IOException;

    @Positive
    public BitArray getUnalignedBitString() throws IOException;

    @Positive
    public byte[] getOctetString() throws IOException;

    @Positive
    public void getNull() throws IOException;

    @Positive
    public ObjectIdentifier getOID() throws IOException;

    @Positive
    public String getUTF8String() throws IOException;

    @Positive
    public String getPrintableString() throws IOException;

    @Positive
    public String getT61String() throws IOException;

    @Positive
    public String getBMPString() throws IOException;

    @Positive
    public String getIA5String() throws IOException;

    @Positive
    public String getGeneralString() throws IOException;

    @Positive
    public Date getUTCTime() throws IOException;

    @Positive
    public Date getGeneralizedTime() throws IOException;

    @Positive
    public DerValue[] getSequence(int startLen) throws IOException;

    @Positive
    public DerValue[] getSet(int startLen) throws IOException;

    @Positive
    public DerValue[] getSet(int startLen, boolean implicit) throws IOException;

    @Positive
    @Pure
    @Positive
    public int peekByte() throws IOException;

    @Positive
    static int getLength(InputStream in) throws IOException;

    @Positive
    static int getDefiniteLength(InputStream in) throws IOException;

    @Positive
    public void mark(int readAheadLimit);

    @Positive
    public void reset();

    @Positive
    public int available();
    @Positive
}
