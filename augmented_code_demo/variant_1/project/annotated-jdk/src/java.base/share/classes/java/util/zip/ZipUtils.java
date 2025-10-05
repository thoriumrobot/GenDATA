/*
    @Positive
 * Copyright (c) 2013, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.file.attribute.FileTime;
    @Positive
import java.time.DateTimeException;
    @Positive
import java.time.Instant;
    @Positive
import java.time.LocalDateTime;
    @Positive
import java.time.ZoneId;
    @Positive
import java.util.Date;
    @Positive
import java.util.concurrent.TimeUnit;
    @Positive
import static java.util.zip.ZipConstants.ENDHDR;
    @Positive
import jdk.internal.misc.Unsafe;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
class ZipUtils {

    @Positive
    public static final long WINDOWS_TIME_NOT_AVAILABLE;

    @Positive
    public static final FileTime winTimeToFileTime(long wtime);

    @Positive
    public static final long fileTimeToWinTime(FileTime ftime);

    @Positive
    @SignedPositive
    @Positive
    public static final long UPPER_UNIXTIME_BOUND;

    @Positive
    public static final FileTime unixTimeToFileTime(long utime);

    @Positive
    public static final long fileTimeToUnixTime(FileTime ftime);

    @Positive
    public static long dosToJavaTime(long dtime);

    @Positive
    public static long extendedDosToJavaTime(long xdostime);

    @Positive
    static long javaToExtendedDosTime(long time);

    @Positive
    static LocalDateTime javaEpochToLocalDateTime(long time);

    @Positive
    public static final int get16(byte[] b, int off);

    @Positive
    public static final long get32(byte[] b, int off);

    @Positive
    public static final long get64(byte[] b, int off);

    @Positive
    public static final int get32S(byte[] b, int off);

    @Positive
    static final int CH(byte[] b, int n);

    @Positive
    static final int SH(byte[] b, int n);

    @Positive
    static final long LG(byte[] b, int n);

    @Positive
    static final long LL(byte[] b, int n);

    @Positive
    static final long GETSIG(byte[] b);

    @Positive
    static final long LOCSIG(byte[] b);

    @Positive
    static final int LOCVER(byte[] b);

    @Positive
    static final int LOCFLG(byte[] b);

    @Positive
    static final int LOCHOW(byte[] b);

    @Positive
    static final long LOCTIM(byte[] b);

    @Positive
    static final long LOCCRC(byte[] b);

    @Positive
    static final long LOCSIZ(byte[] b);

    @Positive
    static final long LOCLEN(byte[] b);

    @Positive
    static final int LOCNAM(byte[] b);

    @Positive
    static final int LOCEXT(byte[] b);

    @Positive
    static final long EXTCRC(byte[] b);

    @Positive
    static final long EXTSIZ(byte[] b);

    @Positive
    static final long EXTLEN(byte[] b);

    @Positive
    static final int ENDSUB(byte[] b);

    @Positive
    static final int ENDTOT(byte[] b);

    @Positive
    static final long ENDSIZ(byte[] b);

    @Positive
    static final long ENDOFF(byte[] b);

    @Positive
    static final int ENDCOM(byte[] b);

    @Positive
    static final int ENDCOM(byte[] b, int off);

    @Positive
    static final long ZIP64_ENDTOD(byte[] b);

    @Positive
    static final long ZIP64_ENDTOT(byte[] b);

    @Positive
    static final long ZIP64_ENDSIZ(byte[] b);

    @Positive
    static final long ZIP64_ENDOFF(byte[] b);

    @Positive
    static final long ZIP64_LOCOFF(byte[] b);

    @Positive
    static final long CENSIG(byte[] b, int pos);

    @Positive
    static final int CENVEM(byte[] b, int pos);

    @Positive
    static final int CENVEM_FA(byte[] b, int pos);

    @Positive
    static final int CENVER(byte[] b, int pos);

    @Positive
    static final int CENFLG(byte[] b, int pos);

    @Positive
    static final int CENHOW(byte[] b, int pos);

    @Positive
    static final long CENTIM(byte[] b, int pos);

    @Positive
    static final long CENCRC(byte[] b, int pos);

    @Positive
    static final long CENSIZ(byte[] b, int pos);

    @Positive
    static final long CENLEN(byte[] b, int pos);

    @Positive
    static final int CENNAM(byte[] b, int pos);

    @Positive
    static final int CENEXT(byte[] b, int pos);

    @Positive
    static final int CENCOM(byte[] b, int pos);

    @Positive
    static final int CENDSK(byte[] b, int pos);

    @Positive
    static final int CENATT(byte[] b, int pos);

    @Positive
    static final long CENATX(byte[] b, int pos);

    @Positive
    static final int CENATX_PERMS(byte[] b, int pos);

    @Positive
    static final long CENOFF(byte[] b, int pos);

    @Positive
    static void loadLibrary();

    @Positive
    static byte[] getBufferArray(ByteBuffer byteBuffer);

    @Positive
    static int getBufferOffset(ByteBuffer byteBuffer);
    @Positive
}

// CFWR semantic augmentation - variant 1
