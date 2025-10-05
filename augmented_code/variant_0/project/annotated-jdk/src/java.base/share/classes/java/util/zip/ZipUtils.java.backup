/*
 * Copyright (c) 2013, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.ByteBuffer;
import java.nio.file.attribute.FileTime;
import java.time.DateTimeException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Date;
import java.util.concurrent.TimeUnit;
import static java.util.zip.ZipConstants.ENDHDR;
import jdk.internal.misc.Unsafe;

@AnnotatedFor({ "signedness" })
class ZipUtils {

    public static final long WINDOWS_TIME_NOT_AVAILABLE;

    public static final FileTime winTimeToFileTime(long wtime);

    public static final long fileTimeToWinTime(FileTime ftime);

    @SignedPositive
    public static final long UPPER_UNIXTIME_BOUND;

    public static final FileTime unixTimeToFileTime(long utime);

    public static final long fileTimeToUnixTime(FileTime ftime);

    public static long dosToJavaTime(long dtime);

    public static long extendedDosToJavaTime(long xdostime);

    static long javaToExtendedDosTime(long time);

    static LocalDateTime javaEpochToLocalDateTime(long time);

    public static final int get16(byte[] b, int off);

    public static final long get32(byte[] b, int off);

    public static final long get64(byte[] b, int off);

    public static final int get32S(byte[] b, int off);

    static final int CH(byte[] b, int n);

    static final int SH(byte[] b, int n);

    static final long LG(byte[] b, int n);

    static final long LL(byte[] b, int n);

    static final long GETSIG(byte[] b);

    static final long LOCSIG(byte[] b);

    static final int LOCVER(byte[] b);

    static final int LOCFLG(byte[] b);

    static final int LOCHOW(byte[] b);

    static final long LOCTIM(byte[] b);

    static final long LOCCRC(byte[] b);

    static final long LOCSIZ(byte[] b);

    static final long LOCLEN(byte[] b);

    static final int LOCNAM(byte[] b);

    static final int LOCEXT(byte[] b);

    static final long EXTCRC(byte[] b);

    static final long EXTSIZ(byte[] b);

    static final long EXTLEN(byte[] b);

    static final int ENDSUB(byte[] b);

    static final int ENDTOT(byte[] b);

    static final long ENDSIZ(byte[] b);

    static final long ENDOFF(byte[] b);

    static final int ENDCOM(byte[] b);

    static final int ENDCOM(byte[] b, int off);

    static final long ZIP64_ENDTOD(byte[] b);

    static final long ZIP64_ENDTOT(byte[] b);

    static final long ZIP64_ENDSIZ(byte[] b);

    static final long ZIP64_ENDOFF(byte[] b);

    static final long ZIP64_LOCOFF(byte[] b);

    static final long CENSIG(byte[] b, int pos);

    static final int CENVEM(byte[] b, int pos);

    static final int CENVEM_FA(byte[] b, int pos);

    static final int CENVER(byte[] b, int pos);

    static final int CENFLG(byte[] b, int pos);

    static final int CENHOW(byte[] b, int pos);

    static final long CENTIM(byte[] b, int pos);

    static final long CENCRC(byte[] b, int pos);

    static final long CENSIZ(byte[] b, int pos);

    static final long CENLEN(byte[] b, int pos);

    static final int CENNAM(byte[] b, int pos);

    static final int CENEXT(byte[] b, int pos);

    static final int CENCOM(byte[] b, int pos);

    static final int CENDSK(byte[] b, int pos);

    static final int CENATT(byte[] b, int pos);

    static final long CENATX(byte[] b, int pos);

    static final int CENATX_PERMS(byte[] b, int pos);

    static final long CENOFF(byte[] b, int pos);

    static void loadLibrary();

    static byte[] getBufferArray(ByteBuffer byteBuffer);

    static int getBufferOffset(ByteBuffer byteBuffer);
}
