/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import sun.nio.cs.UTF_32BE;
    @Positive
import sun.util.calendar.CalendarDate;
    @Positive
import sun.util.calendar.CalendarSystem;
    @Positive
import java.io.*;
    @Positive
import java.math.BigInteger;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.util.*;
    @Positive
import static java.nio.charset.StandardCharsets.*;

    @Positive
public class DerValue {

    @Positive
    @SignedPositive
    @Positive
    public static final byte TAG_UNIVERSAL;

    @Positive
    @SignedPositive
    @Positive
    public static final byte TAG_APPLICATION;

    @Positive
    @SignedPositive
    @Positive
    public static final byte TAG_CONTEXT;

    @Positive
    @SignedPositive
    @Positive
    public static final byte TAG_PRIVATE;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_Boolean;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_Integer;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_BitString;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_OctetString;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_Null;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_ObjectId;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_Enumerated;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_UTF8String;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_PrintableString;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_T61String;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_IA5String;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_UtcTime;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_GeneralizedTime;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_GeneralString;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_UniversalString;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_BMPString;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_Sequence;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_SequenceOf;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_Set;

    @Positive
    @SignedPositive
    @Positive
    public static final byte tag_SetOf;

    @Positive
    public byte tag;

    @Positive
    public final DerInputStream data;

    @Positive
    public boolean isUniversal();

    @Positive
    public boolean isApplication();

    @Positive
    public boolean isContextSpecific();

    @Positive
    public boolean isContextSpecific(byte cntxtTag);

    @Positive
    boolean isPrivate();

    @Positive
    public boolean isConstructed();

    @Positive
    public boolean isConstructed(byte constructedTag);

    @Positive
    public DerValue(String value) {
    @Positive
    }

    @Positive
    public DerValue(byte stringTag, String value) {
    @Positive
    }

    @Positive
    public DerValue(byte tag, byte[] buffer) {
    @Positive
    }

    @Positive
    public static DerValue wrap(byte tag, DerOutputStream out);

    @Positive
    public DerValue(byte[] encoding) throws IOException {
    @Positive
    }

    @Positive
    public DerValue(InputStream in) throws IOException {
    @Positive
    }

    @Positive
    public void encode(DerOutputStream out) throws IOException;

    @Positive
    public final DerInputStream data();

    @Positive
    public final DerInputStream getData();

    @Positive
    public final byte getTag();

    @Positive
    public boolean getBoolean() throws IOException;

    @Positive
    public ObjectIdentifier getOID() throws IOException;

    @Positive
    public byte[] getOctetString() throws IOException;

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
    public String getAsString() throws IOException;

    @Positive
    public byte[] getBitString(boolean tagImplicit) throws IOException;

    @Positive
    public BitArray getUnalignedBitString(boolean tagImplicit) throws IOException;

    @Positive
    public byte[] getDataBytes() throws IOException;

    @Positive
    public String getPrintableString() throws IOException;

    @Positive
    public String getT61String() throws IOException;

    @Positive
    public String getIA5String() throws IOException;

    @Positive
    public String getBMPString() throws IOException;

    @Positive
    public String getUTF8String() throws IOException;

    @Positive
    public String getGeneralString() throws IOException;

    @Positive
    public String getUniversalString() throws IOException;

    @Positive
    public void getNull() throws IOException;

    @Positive
    public Date getUTCTime() throws IOException;

    @Positive
    public Date getGeneralizedTime() throws IOException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public byte[] toByteArray() throws IOException;

    @Positive
    public DerInputStream toDerInputStream() throws IOException;

    @Positive
    public int length();

    @Positive
    public static boolean isPrintableStringChar(char ch);

    @Positive
    public static byte createTag(byte tagClass, boolean form, byte val);

    @Positive
    public void resetTag(byte tag);

    @Positive
    public DerValue withTag(byte newTag);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    DerValue[] subs(byte expectedTag, int startLen) throws IOException;

    @Positive
    public void clear();
    @Positive
}

// CFWR semantic augmentation - variant 1
