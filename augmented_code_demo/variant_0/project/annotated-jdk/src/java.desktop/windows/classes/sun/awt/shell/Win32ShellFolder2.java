/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.awt.shell;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.awt.Image;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.image.AbstractMultiResolutionImage;
    @Positive
import java.awt.image.BufferedImage;
    @Positive
import java.awt.image.ImageObserver;
    @Positive
import java.io.File;
    @Positive
import java.io.FileNotFoundException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Serial;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.concurrent.Callable;
    @Positive
import javax.swing.SwingConstants;

    @Positive
@SuppressWarnings("serial")
    @Positive
final class Win32ShellFolder2 extends ShellFolder {

    @Positive
    public static final int DESKTOP;

    @Positive
    public static final int INTERNET;

    @Positive
    public static final int PROGRAMS;

    @Positive
    public static final int CONTROLS;

    @Positive
    public static final int PRINTERS;

    @Positive
    public static final int PERSONAL;

    @Positive
    public static final int FAVORITES;

    @Positive
    public static final int STARTUP;

    @Positive
    public static final int RECENT;

    @Positive
    public static final int SENDTO;

    @Positive
    public static final int BITBUCKET;

    @Positive
    public static final int STARTMENU;

    @Positive
    public static final int DESKTOPDIRECTORY;

    @Positive
    public static final int DRIVES;

    @Positive
    public static final int NETWORK;

    @Positive
    public static final int NETHOOD;

    @Positive
    public static final int FONTS;

    @Positive
    public static final int TEMPLATES;

    @Positive
    public static final int COMMON_STARTMENU;

    @Positive
    public static final int COMMON_PROGRAMS;

    @Positive
    public static final int COMMON_STARTUP;

    @Positive
    public static final int COMMON_DESKTOPDIRECTORY;

    @Positive
    public static final int APPDATA;

    @Positive
    public static final int PRINTHOOD;

    @Positive
    public static final int ALTSTARTUP;

    @Positive
    public static final int COMMON_ALTSTARTUP;

    @Positive
    public static final int COMMON_FAVORITES;

    @Positive
    public static final int INTERNET_CACHE;

    @Positive
    public static final int COOKIES;

    @Positive
    public static final int HISTORY;

    @Positive
    public static final int ATTRIB_CANCOPY;

    @Positive
    public static final int ATTRIB_CANMOVE;

    @Positive
    public static final int ATTRIB_CANLINK;

    @Positive
    public static final int ATTRIB_CANRENAME;

    @Positive
    public static final int ATTRIB_CANDELETE;

    @Positive
    public static final int ATTRIB_HASPROPSHEET;

    @Positive
    public static final int ATTRIB_DROPTARGET;

    @Positive
    public static final int ATTRIB_LINK;

    @Positive
    public static final int ATTRIB_SHARE;

    @Positive
    public static final int ATTRIB_READONLY;

    @Positive
    public static final int ATTRIB_GHOSTED;

    @Positive
    public static final int ATTRIB_HIDDEN;

    @Positive
    public static final int ATTRIB_FILESYSANCESTOR;

    @Positive
    public static final int ATTRIB_FOLDER;

    @Positive
    public static final int ATTRIB_FILESYSTEM;

    @Positive
    public static final int ATTRIB_HASSUBFOLDER;

    @Positive
    public static final int ATTRIB_VALIDATE;

    @Positive
    public static final int ATTRIB_REMOVABLE;

    @Positive
    public static final int ATTRIB_COMPRESSED;

    @Positive
    public static final int ATTRIB_BROWSABLE;

    @Positive
    public static final int ATTRIB_NONENUMERATED;

    @Positive
    public static final int ATTRIB_NEWCONTENT;

    @Positive
    public static final int SHGDN_NORMAL;

    @Positive
    public static final int SHGDN_INFOLDER;

    @Positive
    public static final int SHGDN_INCLUDE_NONFILESYS;

    @Positive
    public static final int SHGDN_FORADDRESSBAR;

    @Positive
    public static final int SHGDN_FORPARSING;

    @Positive
    public enum SystemIcon {

    @Positive
        IDI_APPLICATION(32512),
    @Positive
        IDI_HAND(32513),
    @Positive
        IDI_ERROR(32513),
    @Positive
        IDI_QUESTION(32514),
    @Positive
        IDI_EXCLAMATION(32515),
    @Positive
        IDI_WARNING(32515),
    @Positive
        IDI_ASTERISK(32516),
    @Positive
        IDI_INFORMATION(32516),
    @Positive
        IDI_WINLOGO(32517);

    @Positive
        public int getIconID();
    @Positive
    }

    @Positive
    static final class KnownFolderDefinition {
    @Positive
    }

    @Positive
    static final class KnownLibraries {
    @Positive
    }

    @Positive
    static class FolderDisposer implements sun.java2d.DisposerRecord {

    @Positive
        public void dispose();
    @Positive
    }

    @Positive
    static Win32ShellFolder2 createShellFolder(Win32ShellFolder2 parent, long pIDL) throws InterruptedException;

    @Positive
    public void setIsPersonal();

    @Positive
    @Serial
    @Positive
    protected Object writeReplace() throws java.io.ObjectStreamException;

    @Positive
    protected void dispose();

    @Positive
    static native long getNextPIDLEntry(long pIDL);

    @Positive
    static native long copyFirstPIDLEntry(long pIDL);

    @Positive
    static native void releasePIDL(long pIDL);

    @Positive
    public long getParentIShellFolder();

    @Positive
    public long getRelativePIDL();

    @Positive
    public Win32ShellFolder2 getDesktop();

    @Positive
    public long getDesktopIShellFolder();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public boolean isFileSystem();

    @Positive
    public boolean hasAttribute(final int attribute);

    @Positive
    static String getFileSystemPath(final int csidl) throws IOException, InterruptedException;

    @Positive
    public File getParentFile();

    @Positive
    public boolean isDirectory();

    @Positive
    public File[] listFiles(final boolean includeHiddenFiles);

    @Positive
    Win32ShellFolder2 getChildByPath(final String filePath) throws InterruptedException;

    @Positive
    public boolean isLink();

    @Positive
    public boolean isHidden();

    @Positive
    public ShellFolder getLinkLocation();

    @Positive
    long parseDisplayName(final String name) throws IOException, InterruptedException;

    @Positive
    public String getDisplayName();

    @Positive
    public String getFolderType();

    @Positive
    public String getExecutableType();

    @Positive
    static native int[] getStandardViewButton0(int iconIndex, boolean small);

    @Positive
    public Image getIcon(final boolean getLargeIcon);

    @Positive
    public Image getIcon(int width, int height);

    @Positive
    static Image getSystemIcon(SystemIcon iconType);

    @Positive
    static Image getShell32Icon(int iconID, int size);

    @Positive
    public File getCanonicalFile() throws IOException;

    @Positive
    public boolean isSpecial();

    @Positive
    public int compareTo(File file2);

    @Positive
    public ShellFolderColumnInfo[] getFolderColumns();

    @Positive
    public Object getFolderColumnValue(final int column);

    @Positive
    boolean isLibrary();

    @Positive
    public void sortChildren(final List<? extends File> files);

    @Positive
    private static class ColumnComparator implements Comparator<File> {

    @Positive
        public ColumnComparator(Win32ShellFolder2 shellFolder, int columnIdx) {
    @Positive
        }

    @Positive
        public int compare(final File o, final File o1);
    @Positive
    }

    @Positive
    static class MultiResolutionIconImage extends AbstractMultiResolutionImage {

    @Positive
        public MultiResolutionIconImage(int baseSize, Map<Integer, Image> resolutionVariants) {
    @Positive
        }

    @Positive
        public MultiResolutionIconImage(int baseSize, Image image) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int getWidth(ImageObserver observer);

    @Positive
        @Override
    @Positive
        public int getHeight(ImageObserver observer);

    @Positive
        @Override
    @Positive
        protected Image getBaseImage();

    @Positive
        @Override
    @Positive
        public Image getResolutionVariant(double width, double height);

    @Positive
        @Override
    @Positive
        public List<Image> getResolutionVariants();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
