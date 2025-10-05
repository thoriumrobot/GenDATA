/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.util;

import org.checkerframework.dataflow.qual.Pure;
import java.io.InputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Date;

public class DerInputStream {

    public DerInputStream(byte[] data, int start, int length, boolean allowBER) {
    }

    public DerInputStream(byte[] data) throws IOException {
    }

    public DerInputStream(byte[] data, int offset, int len) throws IOException {
    }

    public byte[] toByteArray();

    public DerValue getDerValue() throws IOException;

    public int getInteger() throws IOException;

    public BigInteger getBigInteger() throws IOException;

    public BigInteger getPositiveBigInteger() throws IOException;

    public int getEnumerated() throws IOException;

    public byte[] getBitString() throws IOException;

    public BitArray getUnalignedBitString() throws IOException;

    public byte[] getOctetString() throws IOException;

    public void getNull() throws IOException;

    public ObjectIdentifier getOID() throws IOException;

    public String getUTF8String() throws IOException;

    public String getPrintableString() throws IOException;

    public String getT61String() throws IOException;

    public String getBMPString() throws IOException;

    public String getIA5String() throws IOException;

    public String getGeneralString() throws IOException;

    public Date getUTCTime() throws IOException;

    public Date getGeneralizedTime() throws IOException;

    public DerValue[] getSequence(int startLen) throws IOException;

    public DerValue[] getSet(int startLen) throws IOException;

    public DerValue[] getSet(int startLen, boolean implicit) throws IOException;

    @Pure
    public int peekByte() throws IOException;

    static int getLength(InputStream in) throws IOException;

    static int getDefiniteLength(InputStream in) throws IOException;

    public void mark(int readAheadLimit);

    public void reset();

    public int available();
}
