/*
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import static java.util.zip.ZipUtils.*;
import java.nio.file.attribute.FileTime;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.time.LocalDateTime;
import java.time.ZonedDateTime;
import java.time.ZoneId;
import static java.util.zip.ZipConstants64.*;

@AnnotatedFor({ "index", "interning", "nullness", "signedness" })
@UsesObjectEquals
public class ZipEntry implements ZipConstants, Cloneable {

    public static final int STORED;

    public static final int DEFLATED;

    public ZipEntry(String name) {
    }

    public ZipEntry(ZipEntry e) {
    }

    public String getName();

    public void setTime(long time);

    public long getTime();

    public void setTimeLocal(LocalDateTime time);

    public LocalDateTime getTimeLocal();

    public ZipEntry setLastModifiedTime(FileTime time);

    public FileTime getLastModifiedTime();

    public ZipEntry setLastAccessTime(FileTime time);

    public FileTime getLastAccessTime();

    public ZipEntry setCreationTime(FileTime time);

    public FileTime getCreationTime();

    public void setSize(@NonNegative long size);

    @NonNegative
    public long getSize();

    public long getCompressedSize();

    public void setCompressedSize(long csize);

    public void setCrc(long crc);

    public long getCrc();

    public void setMethod(int method);

    public int getMethod();

    public void setExtra(byte[] extra);

    void setExtra0(byte[] extra, boolean doZIP64, boolean isLOC);

    @Pure
    public byte @Nullable [] getExtra();

    public void setComment(String comment);

    @Nullable
    public String getComment();

    public boolean isDirectory();

    public String toString();

    public int hashCode();

    public Object clone();
}
