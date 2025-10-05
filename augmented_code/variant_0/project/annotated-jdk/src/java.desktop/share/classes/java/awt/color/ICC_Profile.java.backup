/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.color;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilePermission;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamException;
import java.io.OutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.StringTokenizer;
import sun.java2d.cmm.CMSManager;
import sun.java2d.cmm.PCMM;
import sun.java2d.cmm.Profile;
import sun.java2d.cmm.ProfileDataVerifier;
import sun.java2d.cmm.ProfileDeferralInfo;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class ICC_Profile implements Serializable {

    private interface BuiltInProfile {

        ICC_Profile SRGB;

        ICC_Profile LRGB;

        ICC_Profile XYZ;

        ICC_Profile PYCC;

        ICC_Profile GRAY;
    }

    public static final int CLASS_INPUT;

    public static final int CLASS_DISPLAY;

    public static final int CLASS_OUTPUT;

    public static final int CLASS_DEVICELINK;

    public static final int CLASS_COLORSPACECONVERSION;

    public static final int CLASS_ABSTRACT;

    public static final int CLASS_NAMEDCOLOR;

    public static final int icSigXYZData;

    public static final int icSigLabData;

    public static final int icSigLuvData;

    public static final int icSigYCbCrData;

    public static final int icSigYxyData;

    public static final int icSigRgbData;

    public static final int icSigGrayData;

    public static final int icSigHsvData;

    public static final int icSigHlsData;

    public static final int icSigCmykData;

    public static final int icSigCmyData;

    public static final int icSigSpace2CLR;

    public static final int icSigSpace3CLR;

    public static final int icSigSpace4CLR;

    public static final int icSigSpace5CLR;

    public static final int icSigSpace6CLR;

    public static final int icSigSpace7CLR;

    public static final int icSigSpace8CLR;

    public static final int icSigSpace9CLR;

    public static final int icSigSpaceACLR;

    public static final int icSigSpaceBCLR;

    public static final int icSigSpaceCCLR;

    public static final int icSigSpaceDCLR;

    public static final int icSigSpaceECLR;

    public static final int icSigSpaceFCLR;

    public static final int icSigInputClass;

    public static final int icSigDisplayClass;

    public static final int icSigOutputClass;

    public static final int icSigLinkClass;

    public static final int icSigAbstractClass;

    public static final int icSigColorSpaceClass;

    public static final int icSigNamedColorClass;

    public static final int icPerceptual;

    public static final int icRelativeColorimetric;

    public static final int icMediaRelativeColorimetric;

    public static final int icSaturation;

    public static final int icAbsoluteColorimetric;

    public static final int icICCAbsoluteColorimetric;

    public static final int icSigHead;

    public static final int icSigAToB0Tag;

    public static final int icSigAToB1Tag;

    public static final int icSigAToB2Tag;

    public static final int icSigBlueColorantTag;

    public static final int icSigBlueMatrixColumnTag;

    public static final int icSigBlueTRCTag;

    public static final int icSigBToA0Tag;

    public static final int icSigBToA1Tag;

    public static final int icSigBToA2Tag;

    public static final int icSigCalibrationDateTimeTag;

    public static final int icSigCharTargetTag;

    public static final int icSigCopyrightTag;

    public static final int icSigCrdInfoTag;

    public static final int icSigDeviceMfgDescTag;

    public static final int icSigDeviceModelDescTag;

    public static final int icSigDeviceSettingsTag;

    public static final int icSigGamutTag;

    public static final int icSigGrayTRCTag;

    public static final int icSigGreenColorantTag;

    public static final int icSigGreenMatrixColumnTag;

    public static final int icSigGreenTRCTag;

    public static final int icSigLuminanceTag;

    public static final int icSigMeasurementTag;

    public static final int icSigMediaBlackPointTag;

    public static final int icSigMediaWhitePointTag;

    public static final int icSigNamedColor2Tag;

    public static final int icSigOutputResponseTag;

    public static final int icSigPreview0Tag;

    public static final int icSigPreview1Tag;

    public static final int icSigPreview2Tag;

    public static final int icSigProfileDescriptionTag;

    public static final int icSigProfileSequenceDescTag;

    public static final int icSigPs2CRD0Tag;

    public static final int icSigPs2CRD1Tag;

    public static final int icSigPs2CRD2Tag;

    public static final int icSigPs2CRD3Tag;

    public static final int icSigPs2CSATag;

    public static final int icSigPs2RenderingIntentTag;

    public static final int icSigRedColorantTag;

    public static final int icSigRedMatrixColumnTag;

    public static final int icSigRedTRCTag;

    public static final int icSigScreeningDescTag;

    public static final int icSigScreeningTag;

    public static final int icSigTechnologyTag;

    public static final int icSigUcrBgTag;

    public static final int icSigViewingCondDescTag;

    public static final int icSigViewingConditionsTag;

    public static final int icSigChromaticityTag;

    public static final int icSigChromaticAdaptationTag;

    public static final int icSigColorantOrderTag;

    public static final int icSigColorantTableTag;

    public static final int icHdrSize;

    public static final int icHdrCmmId;

    public static final int icHdrVersion;

    public static final int icHdrDeviceClass;

    public static final int icHdrColorSpace;

    public static final int icHdrPcs;

    public static final int icHdrDate;

    public static final int icHdrMagic;

    public static final int icHdrPlatform;

    public static final int icHdrFlags;

    public static final int icHdrManufacturer;

    public static final int icHdrModel;

    public static final int icHdrAttributes;

    public static final int icHdrRenderingIntent;

    public static final int icHdrIlluminant;

    public static final int icHdrCreator;

    public static final int icHdrProfileID;

    public static final int icTagType;

    public static final int icTagReserved;

    public static final int icCurveCount;

    public static final int icCurveData;

    public static final int icXYZNumberX;

    @Deprecated()
    @SuppressWarnings("removal")
    protected void finalize();

    public static ICC_Profile getInstance(byte[] data);

    public static ICC_Profile getInstance(int cspace);

    public static ICC_Profile getInstance(String fileName) throws IOException;

    public static ICC_Profile getInstance(InputStream s) throws IOException;

    static byte[] getProfileDataFromStream(InputStream s) throws IOException;

    public int getMajorVersion();

    public int getMinorVersion();

    public int getProfileClass();

    public int getColorSpaceType();

    public int getPCSType();

    public void write(String fileName) throws IOException;

    public void write(OutputStream s) throws IOException;

    public byte[] getData();

    public byte[] getData(int tagSignature);

    public void setData(int tagSignature, byte[] tagData);

    public int getNumComponents();

    float[] getMediaWhitePoint();

    final float[] getXYZTag(int tagSignature);

    float getGamma(int tagSignature);

    short[] getTRC(int tagSignature);

    @Serial
    protected Object readResolve() throws ObjectStreamException;
}
