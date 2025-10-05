/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import static java.util.zip.ZipUtils.*;
    @Positive
import java.nio.file.attribute.FileTime;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.TimeUnit;
    @Positive
import java.time.LocalDateTime;
    @Positive
import java.time.ZonedDateTime;
    @Positive
import java.time.ZoneId;
    @Positive
import static java.util.zip.ZipConstants64.*;

    @Positive
@AnnotatedFor({ "index", "interning", "nullness", "signedness" })
    @Positive
@UsesObjectEquals
    @Positive
public class ZipEntry implements ZipConstants, Cloneable {

    @Positive
    public static final int STORED;

    @Positive
    public static final int DEFLATED;

    @Positive
    public ZipEntry(String name) {
    @Positive
    }

    @Positive
    public ZipEntry(ZipEntry e) {
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    public void setTime(long time);

    @Positive
    public long getTime();

    @Positive
    public void setTimeLocal(LocalDateTime time);

    @Positive
    public LocalDateTime getTimeLocal();

    @Positive
    public ZipEntry setLastModifiedTime(FileTime time);

    @Positive
    public FileTime getLastModifiedTime();

    @Positive
    public ZipEntry setLastAccessTime(FileTime time);

    @Positive
    public FileTime getLastAccessTime();

    @Positive
    public ZipEntry setCreationTime(FileTime time);

    @Positive
    public FileTime getCreationTime();

    @Positive
    public void setSize(@NonNegative long size);

    @Positive
    @NonNegative
    @Positive
    public long getSize();

    @Positive
    public long getCompressedSize();

    @Positive
    public void setCompressedSize(long csize);

    @Positive
    public void setCrc(long crc);

    @Positive
    public long getCrc();

    @Positive
    public void setMethod(int method);

    @Positive
    public int getMethod();

    @Positive
    public void setExtra(byte[] extra);

    @Positive
    void setExtra0(byte[] extra, boolean doZIP64, boolean isLOC);

    @Positive
    @Pure
    @Positive
    public byte @Nullable [] getExtra();

    @Positive
    public void setComment(String comment);

    @Positive
    @Nullable
    @Positive
    public String getComment();

    @Positive
    public boolean isDirectory();

    @Positive
    public String toString();

    @Positive
    public int hashCode();

    @Positive
    public Object clone();
    @Positive
}
